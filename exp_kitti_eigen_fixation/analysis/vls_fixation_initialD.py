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
from exp_kitti_eigen_fixation.eppflowenet.EppFlowNet_scale_initialD import EppFlowNet

from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from PIL import Image, ImageDraw
from core.utils.flow_viz import flow_to_image
from core.utils.utils import InputPadder, forward_interpolate, tensor2disp, tensor2rgb, vls_ins, tensor2grad
from posenet import Posenet
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable

from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))

    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

def concat_imgs(figs):
    w, h = figs[0].size
    left = 0
    # top = h - 220
    # right = 520
    top = h - 352
    right = 1216
    bottom = h
    for k in range(len(figs)):
        figs[k] = figs[k].crop((left, top, right, bottom))

    return figs

@torch.no_grad()
def validate_kitti(model, args, eval_loader, group, isdeepv2d=False):
    """ Peform validation using the KITTI-2015 (train) split """
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    gpu = args.gpu
    eval_measures_depth = torch.zeros(10).cuda(device=gpu)
    vlsroot = '/media/shengjie/disk1/visualization/paper_depthvls'

    residual_root = os.path.join(vlsroot, 'residual_vls')
    bts_root = os.path.join(vlsroot, 'bts_vls')
    ours_root = os.path.join(vlsroot, 'ours_vls')
    deepv2d_root = os.path.join(vlsroot, 'deepv2d_vls')
    rgb_root = os.path.join(vlsroot, 'rgb_in')

    # os.makedirs(residual_root, exist_ok=True)
    # os.makedirs(bts_root, exist_ok=True)
    # os.makedirs(ours_root, exist_ok=True)
    # os.makedirs(deepv2d_root, exist_ok=True)
    # os.makedirs(rgb_root, exist_ok=True)

    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        image1 = data_blob['img1'].cuda(gpu) / 255.0
        image2 = data_blob['img2'].cuda(gpu) / 255.0
        intrinsic = data_blob['intrinsic'].cuda(gpu)
        insmap = data_blob['insmap'].cuda(gpu)
        posepred = data_blob['posepred'].cuda(gpu)
        depthgt = data_blob['depthmap'].cuda(gpu)

        if not args.initbymD:
            mD_pred = data_blob['depthpred'].cuda(gpu)
        else:
            mD_pred = data_blob['mdDepth_pred'].cuda(gpu)

        svname = "{}.png".format(str(val_id).zfill(10))

        mD_pred_clipped = torch.clamp_min(mD_pred, min=args.min_depth_pred)

        outputs = model(image1, image2, mD_pred_clipped, intrinsic, posepred, insmap)
        predread = outputs[('depth', 2)]
        depthpred_deepv2d = data_blob['depthpred_deepv2d'].cuda(gpu)
        sigmoidact = outputs[('residualdepth', 2)]

        # tensor2disp(1 / mD_pred_clipped, vmax=0.15, viewind=0).save(os.path.join(bts_root, svname))
        # tensor2disp(1 / predread, vmax=0.15, viewind=0).save(os.path.join(ours_root, svname))
        # tensor2disp(1 / depthpred_deepv2d, vmax=0.15, viewind=0).save(os.path.join(deepv2d_root, svname))
        # tensor2rgb(image1, viewind=0).save(os.path.join(rgb_root, svname))
        # tensor2grad(sigmoidact, pos_bar=0.1, neg_bar=-0.1, viewind=0).save(os.path.join(residual_root, svname))

        fig1 = tensor2rgb(image1, viewind=0)
        fig1_2 = tensor2rgb(image2, viewind=0)
        fig2 = tensor2disp(1 / depthpred_deepv2d, vmax=0.15, viewind=0)
        fig3 = tensor2disp(1 / mD_pred_clipped, vmax=0.15, viewind=0)
        fig4 = tensor2grad(sigmoidact, pos_bar=0.1, neg_bar=-0.1, viewind=0)
        fig5 = tensor2disp(1 / predread, vmax=0.15, viewind=0)

        figs = concat_imgs([fig1, fig1_2, fig2, fig3, fig4, fig5])

        figc1 = np.concatenate([np.array(figs[0]), np.array(figs[1])], axis=0)
        figc2 = np.concatenate([np.array(figs[4]), np.array(figs[2])], axis=0)
        figc3 = np.concatenate([np.array(figs[3]), np.array(figs[5])], axis=0)
        imgvls = np.concatenate([figc1, figc2, figc3], axis=1)
        imgvls = Image.fromarray(imgvls)

        imgvls.save(os.path.join(vlsroot, svname))

        selector = ((depthgt > 0) * (predread > 0) * (depthgt > args.min_depth_eval) * (depthgt < args.max_depth_eval)).float()
        predread = torch.clamp(predread, min=args.min_depth_eval, max=args.max_depth_eval)
        depth_gt_flatten = depthgt[selector == 1].cpu().numpy()
        pred_depth_flatten = predread[selector == 1].cpu().numpy()

        pred_depth_flatten = np.median(depth_gt_flatten/pred_depth_flatten) * pred_depth_flatten

        eval_measures_depth_np = compute_errors(gt=depth_gt_flatten, pred=pred_depth_flatten)

        eval_measures_depth[:9] += torch.tensor(eval_measures_depth_np).cuda(device=gpu)
        eval_measures_depth[9] += 1

    if args.distributed:
        dist.all_reduce(tensor=eval_measures_depth, op=dist.ReduceOp.SUM, group=group)

    if args.gpu == 0:
        eval_measures_depth[0:9] = eval_measures_depth[0:9] / eval_measures_depth[9]
        eval_measures_depth = eval_measures_depth.cpu().numpy()
        print('Computing Depth errors for %f eval samples' % (eval_measures_depth[9].item()))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
        for i in range(8):
            print('{:7.3f}, '.format(eval_measures_depth[i]), end='')
        print('{:7.3f}'.format(eval_measures_depth[8]))

        return {'silog': float(eval_measures_depth[0]),
                'abs_rel': float(eval_measures_depth[1]),
                'log10': float(eval_measures_depth[2]),
                'rms': float(eval_measures_depth[3]),
                'sq_rel': float(eval_measures_depth[4]),
                'log_rms': float(eval_measures_depth[5]),
                'd1': float(eval_measures_depth[6]),
                'd2': float(eval_measures_depth[7]),
                'd3': float(eval_measures_depth[8])
                }
    else:
        return None

MAX_FLOW = 400

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

    if args.restore_ckpt is not None:
        print("=> loading checkpoint '{}'".format(args.restore_ckpt))
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.restore_ckpt, map_location=loc)
        model.load_state_dict(checkpoint, strict=False)

    train_entries, evaluation_entries = read_splits()

    eval_dataset = KITTI_eigen(root=args.dataset_root, inheight=args.evalheight, inwidth=args.evalwidth, entries=evaluation_entries, maxinsnum=args.maxinsnum,
                               depth_root=args.depth_root, depthvls_root=args.depthvlsgt_root, prediction_root=args.prediction_root, deepv2dpred_root=args.deepv2dpred_root,
                               mdPred_root=args.mdPred_root, ins_root=args.ins_root, istrain=False, isgarg=True, RANSACPose_root=args.RANSACPose_root, baninsmap=args.baninsmap)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset) if args.distributed else None
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=True, num_workers=3, drop_last=True, sampler=eval_sampler)

    print("Test splits contain %d images" % (eval_dataset.__len__()))

    if args.distributed:
        group = dist.new_group([i for i in range(ngpus_per_node)])

    validate_kitti(model.module, args, eval_loader, group, isdeepv2d=False)
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
    parser.add_argument('--baninsmap', action='store_true')

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--semantics_root', type=str)
    parser.add_argument('--depth_root', type=str)
    parser.add_argument('--depthvlsgt_root', type=str)
    parser.add_argument('--prediction_root', type=str, default=None)
    parser.add_argument('--deepv2dpred_root', type=str)
    parser.add_argument('--mdPred_root', type=str)
    parser.add_argument('--initbymD', action='store_true')
    parser.add_argument('--ins_root', type=str)
    parser.add_argument('--logroot', type=str)
    parser.add_argument('--RANSACPose_root', type=str)
    parser.add_argument('--num_workers', type=int, default=12)

    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dist_url', type=str, help='url used to set up distributed training', default='tcp://127.0.0.1:1235')
    parser.add_argument('--dist_backend', type=str, help='distributed backend', default='nccl')

    args = parser.parse_args()
    args.dist_url = args.dist_url.rstrip('1235') + str(np.random.randint(2000, 3000, 1).item())

    torch.manual_seed(1234)
    np.random.seed(1234)

    torch.cuda.empty_cache()

    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        args.world_size = ngpus_per_node
        mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        train(args.gpu, ngpus_per_node, args)