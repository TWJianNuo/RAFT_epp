from __future__ import print_function, division
import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import pickle

from torch.utils.data import DataLoader
from exp_kitti_eigen.dataset_kitti_eigen import KITTI_eigen

from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from PIL import Image, ImageDraw
from core.utils.flow_viz import flow_to_image
from core.raft import RAFT
from core.utils.utils import InputPadder, forward_interpolate
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable

from eppcore.eppcore_py import EPPCore

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


def readlines(filename):
    with open(filename, 'r') as f:
        filenames = f.readlines()
    return filenames

def read_deepv2d_pose(entry):
    # Read Pose from Deepv2d
    exportroot = '/media/shengjie/disk1/Prediction/Deepv2d_eigen'
    seq, frameidx, _ = entry.split(' ')
    posesstr = readlines(os.path.join(exportroot, seq, "posepred/{}.txt".format(str(frameidx).zfill(10))))
    poses = list()
    for pstr in posesstr:
        pose = np.zeros([4, 4]).flatten()
        for idx, ele in enumerate(pstr.split(' ')):
            pose[idx] = float(ele)
            if idx == 15:
                break
        pose = np.reshape(pose, [4, 4])
        poses.append(pose)
    pose_deepv2d = poses[3] @ np.linalg.inv(poses[0])
    pose_deepv2d[0:3, 3] = pose_deepv2d[0:3, 3] * 10
    return torch.from_numpy(pose_deepv2d).float()

@torch.no_grad()
def validate_kitti(model, args, eval_loader, eppCbck, eppc_dict, group, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    epe_list = torch.zeros(2).cuda(device=args.gpu)
    out_list = torch.zeros(2).cuda(device=args.gpu)
    eppc_list = torch.zeros(2).cuda(device=args.gpu)

    mvl_list = torch.zeros(2).cuda(device=args.gpu)
    angl_list = torch.zeros(2).cuda(device=args.gpu)
    for val_id, batch in enumerate(tqdm(eval_loader)):
        image1 = batch['img1']
        image1 = Variable(image1)
        image1 = image1.cuda(args.gpu, non_blocking=True)

        image2 = batch['img2']
        image2 = Variable(image2)
        image2 = image2.cuda(args.gpu, non_blocking=True)

        flow_gt = batch['flow']
        flow_gt = Variable(flow_gt)
        flow_gt = flow_gt.cuda(args.gpu, non_blocking=True)
        flow_gt = flow_gt[0]

        valid_gt = batch['valid']
        valid_gt = Variable(valid_gt)
        valid_gt = valid_gt.cuda(args.gpu, non_blocking=True)
        valid_gt = valid_gt[0]

        intrinsic = batch['intrinsic']
        intrinsic = Variable(intrinsic)
        intrinsic = intrinsic.cuda(args.gpu, non_blocking=True)

        rel_pose = batch['rel_pose']
        rel_pose = Variable(rel_pose)
        rel_pose = rel_pose.cuda(args.gpu, non_blocking=True)

        E = batch['E']
        E = Variable(E)
        E = E.cuda(args.gpu, non_blocking=True)

        semantic_selector = batch['semantic_selector']
        semantic_selector = Variable(semantic_selector)
        semantic_selector = semantic_selector.cuda(args.gpu, non_blocking=True)

        depth = batch['depth']
        depth = Variable(depth)
        depth = depth.cuda(args.gpu, non_blocking=True)

        rel_pose_deepv2d = read_deepv2d_pose(batch['entry'][0])
        rel_pose_deepv2d = Variable(rel_pose_deepv2d)
        rel_pose_deepv2d = rel_pose_deepv2d.cuda(args.gpu, non_blocking=True)

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0])

        bz, _, h, w = depth.shape
        eppc_key = "{}_{}".format(h, w)
        if eppc_key not in eppc_dict:
            eppc = EPPCore(bz=1, height=h, width=w, itnum=100, lr=0.1, lap=1e-2, maxinsnum=10)
            eppc.to(f'cuda:{args.gpu}')
            eppc_dict[eppc_key] = eppc
        eppc = eppc_dict[eppc_key]

        instancemap = torch.zeros_like(depth)
        instancemap[semantic_selector.unsqueeze(0) == 0] = -1
        instancemap = instancemap.int()
        t, R = eppc.flow2epp(insmap=instancemap, flowmap=flow.unsqueeze(0), intrinsic=intrinsic)
        ang = eppc.R2ang(R)

        t_est = t[0,0].squeeze()
        ang_est = ang[0,0].squeeze()

        t_gt = rel_pose[0, 0:3, 3] / torch.norm(rel_pose[0, 0:3, 3])
        ang_gt = (eppc.R2ang(rel_pose[0, 0:3, 0:3].unsqueeze(0).unsqueeze(0).expand([-1, 10, -1, -1]))[0,0]).squeeze()

        loss_mv = 1 - torch.sum(t_est * t_gt)
        loss_ang = (ang_est - ang_gt).abs().mean()

        if loss_mv > 0.1:
            continue

        eppc = eppCbck.epp_constrain_val(flowest=flow, E=E, valid=semantic_selector)
        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()

        epe_list[0] += epe[val].mean().item()
        epe_list[1] += 1

        out_list[0] += out[val].sum()
        out_list[1] += torch.sum(val)

        eppc_list[0] += eppc
        eppc_list[1] += 1

        mvl_list[0] += loss_mv
        mvl_list[1] += 1

        angl_list[0] += loss_ang
        angl_list[1] += 1

    if args.distributed:
        dist.all_reduce(tensor=epe_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=out_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=eppc_list, op=dist.ReduceOp.SUM, group=group)

        dist.all_reduce(tensor=mvl_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=angl_list, op=dist.ReduceOp.SUM, group=group)

    if args.gpu == 0:
        epe = epe_list[0] / epe_list[1]
        f1 = 100 * out_list[0] / out_list[1]
        eppc = eppc_list[0] / eppc_list[1]

        mvl = mvl_list[0] / mvl_list[1]
        angl = angl_list[0] / angl_list[1]

        return {'kitti-epe': float(epe.detach().cpu().numpy()), 'kitti-f1': float(f1.detach().cpu().numpy()), 'kitti-eppc': float(eppc.detach().cpu().numpy()),
                'mvl': float(mvl.detach().cpu().numpy()), 'angl': float(angl.detach().cpu().numpy())
                }
    else:
        return None

def read_splits():
    split_root = os.path.join(project_rootdir, 'exp_kitti_eigen', 'splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'train_files.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'test_files.txt'), 'r')]
    return train_entries, evaluation_entries

class eppConstrainer_background(torch.nn.Module):
    def __init__(self, height, width, bz):
        super(eppConstrainer_background, self).__init__()

        self.height = height
        self.width = width
        self.bz = bz

        xx, yy = np.meshgrid(range(width), range(height))
        pts = np.stack([xx, yy, np.ones_like(xx)], axis=2)
        self.pts = nn.Parameter(torch.from_numpy(pts).float().unsqueeze(0).unsqueeze(4).repeat([self.bz, 1, 1, 1, 1]).contiguous(), requires_grad=False)

        self.pts_dict_eval = dict()

    def epp_constrain(self, flowest, E, valid):
        bz, _, h, w = flowest.shape
        assert (self.height == h) and (self.width == w)

        flowest_ex = torch.cat([flowest, torch.zeros([bz, 1, h, w], device=flowest.device, dtype=flowest.dtype)], dim=1).permute([0, 2, 3, 1]).unsqueeze(4)

        ptse1 = self.pts
        ptse2 = self.pts + flowest_ex

        Ee = E.unsqueeze(1).unsqueeze(1).expand([-1, h, w, -1, -1])

        eppcons = torch.sum(ptse2.transpose(dim0=3, dim1=4) @ Ee * ptse1.transpose(dim0=3, dim1=4), dim=[3, 4])
        eppcons_loss = torch.sum(torch.abs(eppcons) * valid) / (torch.sum(valid) + 1)

        return eppcons_loss

    def epp_constrain_val(self, flowest, E, valid):
        if len(flowest.shape) == 3:
            flowest = flowest.unsqueeze(0)
        bz, _, h, w = flowest.shape
        dictkey = "{}_{}".format(h, w)

        if dictkey not in self.pts_dict_eval.keys():
            xx, yy = np.meshgrid(range(w), range(h))
            pts = np.stack([xx, yy, np.ones_like(xx)], axis=2)
            pts = torch.from_numpy(pts).float().unsqueeze(0).unsqueeze(4).cuda(flowest.device)
            self.pts_dict_eval[dictkey] = pts
        else:
            pts = self.pts_dict_eval[dictkey]

        flowest_ex = torch.cat([flowest, torch.zeros([bz, 1, h, w], device=flowest.device, dtype=flowest.dtype)], dim=1).permute([0, 2, 3, 1]).unsqueeze(4)

        ptse1 = pts.expand([bz, -1, -1, -1, -1])
        ptse2 = ptse1 + flowest_ex

        Ee = E.unsqueeze(1).unsqueeze(1).expand([-1, h, w, -1, -1])

        eppcons = torch.sum(ptse2.transpose(dim0=3, dim1=4) @ Ee * ptse1.transpose(dim0=3, dim1=4), dim=[3, 4])
        eppcons_loss = torch.sum(torch.abs(eppcons) * valid) / (torch.sum(valid) + 1)

        return eppcons_loss

def train(gpu, ngpus_per_node, args):
    print("Using GPU %d for training" % gpu)
    args.gpu = gpu

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=ngpus_per_node, rank=args.gpu)

    model = RAFT(args)
    eppc_dict = dict()
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(module=model)
        model = model.to(f'cuda:{args.gpu}')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True, output_device=args.gpu)

        eppCbck = eppConstrainer_background(height=args.image_size[0], width=args.image_size[1], bz=args.batch_size)
        eppCbck.to(f'cuda:{args.gpu}')
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()


    if args.restore_ckpt is not None:
        print("=> loading checkpoint '{}'".format(args.restore_ckpt))
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.restore_ckpt, map_location=loc)
        model.load_state_dict(checkpoint, strict=False)

    model.eval()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    _, evaluation_entries = read_splits()

    eval_dataset = KITTI_eigen(split='evaluation', root=args.dataset_root, entries=evaluation_entries, semantics_root=args.semantics_root, depth_root=args.depth_root)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset) if args.distributed else None
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=True,
                                   shuffle=(eval_sampler is None), num_workers=4, drop_last=True,
                                   sampler=eval_sampler)

    if args.distributed:
        group = dist.new_group([i for i in range(ngpus_per_node)])

    print(validate_kitti(model.module, args, eval_loader, eppCbck, eppc_dict, group))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--stage', help="determines which dataset to use for training", default='kitti')
    parser.add_argument('--small', action='store_true', help='use small model')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--semantics_root', type=str)
    parser.add_argument('--depth_root', type=str)
    parser.add_argument('--logroot', type=str)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--eppcw', type=float, default=0.1)

    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dist_url', type=str, help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
    parser.add_argument('--dist_backend', type=str, help='distributed backend', default='nccl')

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    torch.cuda.empty_cache()

    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        args.world_size = ngpus_per_node
        mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        train(args.gpu, ngpus_per_node, args)