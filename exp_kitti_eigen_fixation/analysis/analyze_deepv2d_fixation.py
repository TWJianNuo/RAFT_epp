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
from core.utils.utils import InputPadder, forward_interpolate, tensor2disp, tensor2rgb, vls_ins, tensor2grad
from posenet import Posenet
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable

from tqdm import tqdm

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def get_vls_eval(data_blob, outputs, args):
    img1 = data_blob['img1'][0].permute([1, 2, 0]).numpy().astype(np.uint8)
    insmap = data_blob['insmap'][0].squeeze().numpy()

    insvls = vls_ins(img1, insmap)

    depthpredvls = tensor2disp(1 / outputs[('depth', 2)], vmax=0.15, viewind=0)
    depthpredvls_org = tensor2disp(1 / data_blob['depthpred'], vmax=0.15, viewind=0)

    sigmoidact = outputs[('residualdepth', 2)]
    sigmoidactvls = tensor2grad(sigmoidact, pos_bar=0.1, neg_bar=-0.1, viewind=0)

    img_val_up = np.concatenate([np.array(insvls), np.array(sigmoidactvls)], axis=1)
    img_val_mid2 = np.concatenate([np.array(depthpredvls), np.array(depthpredvls_org)], axis=1)
    img_val = np.concatenate([np.array(img_val_up), np.array(img_val_mid2)], axis=0)

    svtag = data_blob['tag'][0]
    svroot = os.path.join(args.svroot, args.restore_ckpt.split('/')[-2])
    os.makedirs(svroot, exist_ok=True)

    seq, imgname = svtag.split(' ')
    figname = "{}_{}".format(seq.split('/')[-1], imgname)
    vlsname = os.path.join(svroot, "{}.png".format(figname))
    Image.fromarray(img_val).save(vlsname)

@torch.no_grad()
def validate_kitti(model, args, eval_loader, group, isdeepv2dpred=False, isbackground=True, isvls=True, iters=2):
    """ Peform validation using the KITTI-2015 (train) split """
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    gpu = args.gpu
    eval_reld = torch.zeros(2).cuda(device=gpu)
    eval_measures_depth = torch.zeros(10).cuda(device=gpu)

    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        image1 = data_blob['img1'].cuda(gpu) / 255.0
        image2 = data_blob['img2'].cuda(gpu) / 255.0
        intrinsic = data_blob['intrinsic'].cuda(gpu)
        insmap = data_blob['insmap'].cuda(gpu)
        depthpred = data_blob['depthpred'].cuda(gpu)
        posepred = data_blob['posepred'].cuda(gpu)
        selfpose_gt = data_blob['rel_pose'].cuda(gpu)
        depthgt = data_blob['depthmap'].cuda(gpu)

        reldepth_gt = torch.log(depthgt + 1e-10) - torch.log(torch.sqrt(torch.sum(selfpose_gt[:, 0:3, 3] ** 2, dim=1, keepdim=True))).unsqueeze(-1).unsqueeze(-1).expand([-1, -1, args.evalheight, args.evalwidth])

        for k in range(iters):
            outputs = model(image1, image2, depthpred, intrinsic, posepred, insmap)
            residual_logdepth = outputs[('residualdepth', 2)]

            scalemask = ((insmap == 0) * (torch.sum(outputs[('reconImg', 2)], dim=1, keepdim=True) > 0)).float()
            residual_log = torch.sum(residual_logdepth * scalemask) / torch.sum(scalemask)
            posepred[0, 0, 0:3, 3] = posepred[0, 0, 0:3, 3] * torch.exp(-residual_log)

        if isdeepv2dpred:
            predreld = outputs[('relativedepth', 2)]
            predread = outputs[('depth', 2)]
        else:
            predreld = outputs[('org_relativedepth', 2)]
            predread = depthpred

        if isbackground:
            selector = ((depthgt > 0) * (insmap == 0) * (predread > 0)).float()
        else:
            selector = ((depthgt > 0) * (insmap > 0) * (predread > 0)).float()

        if torch.sum(selector) > 0:
            depthloss = torch.sum(torch.abs(predreld - reldepth_gt) * selector) / (torch.sum(selector) + 1)

            eval_reld[0] += depthloss
            eval_reld[1] += 1

            depth_gt_flatten = depthgt[selector == 1].cpu().numpy()
            pred_depth_flatten = predread[selector == 1].cpu().numpy()
            eval_measures_depth_np = compute_errors(gt=depth_gt_flatten, pred=pred_depth_flatten)
            eval_measures_depth[:9] += torch.tensor(eval_measures_depth_np).cuda(device=gpu)
            eval_measures_depth[9] += 1

        if isvls:
            get_vls_eval(data_blob, outputs, args)

    if args.distributed:
        dist.all_reduce(tensor=eval_reld, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=eval_measures_depth, op=dist.ReduceOp.SUM, group=group)

    if args.gpu == 0:
        eval_reld[0] = eval_reld[0] / eval_reld[1]
        eval_measures_depth[0:9] = eval_measures_depth[0:9] / eval_measures_depth[9]

        eval_measures_depth = eval_measures_depth.cpu().numpy()
        eval_reld = eval_reld.cpu().numpy()

        if isdeepv2dpred:
            predsrc = 'Deepv2d'
        else:
            predsrc = 'MonocularDepth'

        if isbackground:
            evalcomponent = 'Background'
        else:
            evalcomponent = 'Foreground'

        print('Computing Depth errors for {} eval samples from {} on {}'.format(eval_reld[1].item(), predsrc, evalcomponent))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3', 'reld'))
        for i in range(9):
            print('{:7.3f}, '.format(eval_measures_depth[i]), end='')
        print('{:7.3f}'.format(eval_reld[0]))

        return {'silog': float(eval_measures_depth[0]),
                'abs_rel': float(eval_measures_depth[1]),
                'log10': float(eval_measures_depth[2]),
                'rms': float(eval_measures_depth[3]),
                'sq_rel': float(eval_measures_depth[4]),
                'log_rms': float(eval_measures_depth[5]),
                'd1': float(eval_measures_depth[6]),
                'd2': float(eval_measures_depth[7]),
                'd3': float(eval_measures_depth[8]),
                'reld': float(eval_reld[0])
                }
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

    model.eval()

    train_entries, evaluation_entries = read_splits()

    eval_dataset = KITTI_eigen(root=args.dataset_root, inheight=args.evalheight, inwidth=args.evalwidth, entries=evaluation_entries, maxinsnum=args.maxinsnum,
                               depth_root=args.depth_root, depthvls_root=args.depthvlsgt_root, prediction_root=args.prediction_root, ins_root=args.ins_root, istrain=False)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset) if args.distributed else None
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=True, num_workers=3, drop_last=True, sampler=eval_sampler)

    print("Test splits contain %d images" % (eval_dataset.__len__()))

    if args.distributed:
        group = dist.new_group([i for i in range(ngpus_per_node)])

    validate_kitti(model.module, args, eval_loader, group, isdeepv2dpred=True, isbackground=True, isvls=False)
    # validate_kitti(model.module, args, eval_loader, group, isdeepv2dpred=False, isbackground=True, isvls=False)
    # validate_kitti(model.module, args, eval_loader, group, isdeepv2dpred=True, isbackground=False, isvls=False)
    # validate_kitti(model.module, args, eval_loader, group, isdeepv2dpred=False, isbackground=False, isvls=False)
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
    parser.add_argument('--svroot', type=str)

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