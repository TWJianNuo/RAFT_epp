from __future__ import print_function, division
import os, sys
project_rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader
from exp_kitti_eigen_fixation.dataset_kitti_eigen_fixation import KITTI_eigen
from exp_kitti_eigen_fixation.eppflowenet.EppFlowNet import EppFlowNet

from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from PIL import Image, ImageDraw
from core.utils.flow_viz import flow_to_image
from core.utils.utils import InputPadder, forward_interpolate, tensor2disp, tensor2rgb, vls_ins
from posenet import Posenet
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable

from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_kitti(model, args, eval_loader, group, isdeepv2dpred=False):
    """ Peform validation using the KITTI-2015 (train) split """
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    gpu = args.gpu
    eval_reld = torch.zeros(2).cuda(device=gpu)
    eval_relpose = torch.zeros(2).cuda(device=gpu)

    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        image1 = data_blob['img1'].cuda(gpu) / 255.0
        image2 = data_blob['img2'].cuda(gpu) / 255.0
        intrinsic = data_blob['intrinsic'].cuda(gpu)
        insmap = data_blob['insmap'].cuda(gpu)
        mD_pred = data_blob['depthpred'].cuda(gpu)
        aposes_pred = data_blob['posepred'].cuda(gpu)
        sposes_gt = data_blob['rel_pose'].cuda(gpu)
        depthgt = data_blob['depthmap'].cuda(gpu)
        depthpred_deepv2d = data_blob['depthpred_deepv2d'].cuda(gpu)
        posepred_deepv2d = data_blob['posepred_deepv2d'].cuda(gpu)

        # Relative Logged Depth is defined as log(depth) - log(self_scale)
        sposes_gt_scale = torch.log(torch.sqrt(torch.sum(sposes_gt[:, 0:3, 3] ** 2, dim=1)))
        aposes_pred_scale = torch.log(torch.sqrt(torch.sum(aposes_pred[:, 0, 0:3, 3] ** 2, dim=1)))

        rl_depthgt = torch.log(depthgt + 1e-10) - sposes_gt_scale.view([1, 1, 1, 1]).expand([-1, -1, args.evalheight, args.evalwidth])

        outputs = model(image1, image2, mD_pred, intrinsic, aposes_pred, insmap)

        if isdeepv2dpred:
            predreld = outputs[('relativedepth', 2)]
            scaleloss_selector = ((torch.sum(outputs[('reconImg', 2)], dim=1, keepdim=True) > 0) * (insmap == 0) * (mD_pred > 0) * (depthpred_deepv2d > 0)).float()
            resd_pred = -torch.sum(outputs[('residualdepth', 2)] * scaleloss_selector, dim=[1, 2, 3]) / (torch.sum(scaleloss_selector, dim=[1, 2, 3]) + 1)
        else:
            predreld = outputs[('org_relativedepth', 2)]
            resd_pred = 0

        resscaleloss = torch.abs(aposes_pred_scale + resd_pred - sposes_gt_scale).sum()

        selector = ((depthgt > 0) * (insmap == 0) * (mD_pred > 0) * (depthpred_deepv2d > 0)).float()
        depthloss = (torch.sum(torch.abs(predreld - rl_depthgt) * selector, dim=[1, 2, 3]) / (torch.sum(selector, dim=[1, 2, 3]) + 1)).sum()

        eval_reld[0] += depthloss
        eval_reld[1] += 1

        eval_relpose[0] += resscaleloss
        eval_relpose[1] += 1

    if args.distributed:
        dist.all_reduce(tensor=eval_reld, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=eval_relpose, op=dist.ReduceOp.SUM, group=group)

    if args.gpu == 0:
        eval_reld[0] = eval_reld[0] / eval_reld[1]
        eval_relpose[0] = eval_relpose[0] / eval_relpose[1]

        print("in {} eval samples: Pose scale Loss: {:7.3f}, Absolute Relative Depth Loss: {:7.3f}".format(eval_reld[1].item(), eval_relpose[0].item(), eval_reld[0].item()))
        return {'reld': float(eval_reld[0].item()), 'relpose': float(eval_relpose[0].item())}
    else:
        return None

@torch.no_grad()
def validate_kitti_deepv2d(model, args, eval_loader, group):
    """ Peform validation using the KITTI-2015 (train) split """
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    gpu = args.gpu
    eval_reld = torch.zeros(2).cuda(device=gpu)
    eval_relpose = torch.zeros(2).cuda(device=gpu)

    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        image1 = data_blob['img1'].cuda(gpu) / 255.0
        image2 = data_blob['img2'].cuda(gpu) / 255.0
        intrinsic = data_blob['intrinsic'].cuda(gpu)
        insmap = data_blob['insmap'].cuda(gpu)
        mD_pred = data_blob['depthpred'].cuda(gpu)
        aposes_pred = data_blob['posepred'].cuda(gpu)
        sposes_gt = data_blob['rel_pose'].cuda(gpu)
        depthgt = data_blob['depthmap'].cuda(gpu)
        depthpred_deepv2d = data_blob['depthpred_deepv2d'].cuda(gpu)
        posepred_deepv2d = data_blob['posepred_deepv2d'].cuda(gpu)

        # Relative Logged Depth is defined as log(depth) - log(self_scale)
        sposes_gt_scale = torch.log(torch.sqrt(torch.sum(sposes_gt[:, 0:3, 3] ** 2, dim=1)))
        aposes_pred_scale = torch.log(torch.sqrt(torch.sum(posepred_deepv2d[:, 0, 0:3, 3] ** 2, dim=1)))

        rl_depthgt = torch.log(depthgt + 1e-10) - sposes_gt_scale.view([1, 1, 1, 1]).expand([-1, -1, args.evalheight, args.evalwidth])

        outputs = model(image1, image2, mD_pred, intrinsic, aposes_pred, insmap)

        predreld = outputs[('org_relativedepth', 2)]
        resd_pred = 0

        resscaleloss = torch.abs(aposes_pred_scale + resd_pred - sposes_gt_scale).sum()

        selector = ((depthgt > 0) * (insmap == 0) * (mD_pred > 0) * (depthpred_deepv2d > 0)).float()
        depthloss = (torch.sum(torch.abs(predreld - rl_depthgt) * selector, dim=[1, 2, 3]) / (torch.sum(selector, dim=[1, 2, 3]) + 1)).sum()

        eval_reld[0] += depthloss
        eval_reld[1] += 1

        eval_relpose[0] += resscaleloss
        eval_relpose[1] += 1

    if args.distributed:
        dist.all_reduce(tensor=eval_reld, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=eval_relpose, op=dist.ReduceOp.SUM, group=group)

    if args.gpu == 0:
        eval_reld[0] = eval_reld[0] / eval_reld[1]
        eval_relpose[0] = eval_relpose[0] / eval_relpose[1]

        print("in {} eval samples: Pose scale Loss: {:7.3f}, Absolute Relative Depth Loss: {:7.3f}".format(eval_reld[1].item(), eval_relpose[0].item(), eval_reld[0].item()))
        return {'reld': float(eval_reld[0].item()), 'relpose': float(eval_relpose[0].item())}
    else:
        return None


def read_splits():
    split_root = os.path.join(project_rootdir, 'exp_pose_mdepth_kitti_eigen/splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'train_files.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'test_files.txt'), 'r')]
    return train_entries, evaluation_entries

def train(gpu, ngpus_per_node, args):
    print("Using GPU %d for training" % gpu)
    args.gpu = gpu

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=ngpus_per_node, rank=args.gpu)

    model = EppFlowNet(args=args)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(module=model)
        model = model.to(f'cuda:{args.gpu}')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True, output_device=args.gpu)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    logroot = os.path.join(args.logroot, args.name)
    print("Parameter Count: %d, saving location: %s" % (count_parameters(model), logroot))

    if args.restore_ckpt is not None:
        print("=> loading checkpoint '{}'".format(args.restore_ckpt))
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.restore_ckpt, map_location=loc)
        model.load_state_dict(checkpoint, strict=False)

    model.train()

    train_entries, evaluation_entries = read_splits()

    eval_dataset = KITTI_eigen(root=args.dataset_root, inheight=args.evalheight, inwidth=args.evalwidth, entries=evaluation_entries, maxinsnum=args.maxinsnum,
                               depth_root=args.depth_root, depthvls_root=args.depthvlsgt_root, prediction_root=args.prediction_root, ins_root=args.ins_root, istrain=False, isgarg=True,
                               deepv2dpred_root=args.deepv2dpred_root)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset) if args.distributed else None
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=True, num_workers=3, drop_last=True, sampler=eval_sampler)

    print("Test splits contain %d images" % (eval_dataset.__len__()))

    if args.distributed:
        group = dist.new_group([i for i in range(ngpus_per_node)])

    validate_kitti_deepv2d(model.module, args, eval_loader, group)
    validate_kitti(model.module, args, eval_loader, group, isdeepv2dpred=True)
    validate_kitti(model.module, args, eval_loader, group, isdeepv2dpred=False)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--inheight', type=int, default=320)
    parser.add_argument('--inwidth', type=int, default=960)
    parser.add_argument('--evalheight', type=int, default=320)
    parser.add_argument('--evalwidth', type=int, default=1216)
    parser.add_argument('--maxinsnum', type=int, default=50)
    parser.add_argument('--min_depth_pred', type=float, default=1)
    parser.add_argument('--max_depth_pred', type=float, default=85)
    parser.add_argument('--min_depth_eval', type=float, default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, default=80)

    parser.add_argument('--tscale_range', type=float, default=3)
    parser.add_argument('--objtscale_range', type=float, default=10)
    parser.add_argument('--angx_range', type=float, default=0.03)
    parser.add_argument('--angy_range', type=float, default=0.06)
    parser.add_argument('--angz_range', type=float, default=0.01)
    parser.add_argument('--num_layers', type=int, default=50)
    parser.add_argument('--num_deges', type=int, default=32)
    parser.add_argument('--maxlogscale', type=float, default=1.5)

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--semantics_root', type=str)
    parser.add_argument('--depth_root', type=str)
    parser.add_argument('--depthvlsgt_root', type=str)
    parser.add_argument('--prediction_root', type=str)
    parser.add_argument('--deepv2dpred_root', type=str)
    parser.add_argument('--ins_root', type=str)
    parser.add_argument('--logroot', type=str)
    parser.add_argument('--num_workers', type=int, default=12)

    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dist_url', type=str, help='url used to set up distributed training', default='tcp://127.0.0.1:1235')
    parser.add_argument('--dist_backend', type=str, help='distributed backend', default='nccl')

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir(os.path.join(args.logroot, args.name)):
        os.makedirs(os.path.join(args.logroot, args.name), exist_ok=True)
    os.makedirs(os.path.join(args.logroot, 'evaluation', args.name), exist_ok=True)

    torch.cuda.empty_cache()

    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        args.world_size = ngpus_per_node
        mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        train(args.gpu, ngpus_per_node, args)