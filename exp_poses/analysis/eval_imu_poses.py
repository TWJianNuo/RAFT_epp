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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader
from exp_poses.dataset_kitti_stereo15_orged import KITTI_eigen_stereo15
from exp_kitti_eigen_fixation.eppflowenet.EppFlowNet_scratch import EppFlowNet

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

MAX_FLOW = 400

def vls_flows(image1, image2, flow_anno, flow_depth, depth, insmap):
    image1np = image1[0].cpu().permute([1, 2, 0]).numpy().astype(np.uint8)
    image2np = image2[0].cpu().permute([1, 2, 0]).numpy().astype(np.uint8)
    depthnp = depth[0].cpu().squeeze().numpy()
    flow_anno_np = flow_anno[0].cpu().numpy()
    flow_depth_np = flow_depth[0].cpu().numpy()
    insmap_np = insmap[0].cpu().squeeze().numpy()

    # tensor2disp(depth > 0, vmax=1, viewind=0).show()
    # tensor2disp(flow_anno[:, 1:2, :, :] != 0, vmax=1, viewind=0).show()

    h, w, _ = image1np.shape
    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
    selector_anno = (flow_anno_np[0, :, :] != 0) * (depthnp > 0) * (insmap_np == 0)

    flowx = flow_anno_np[0][selector_anno]
    flowy = flow_anno_np[1][selector_anno]

    xxf = xx[selector_anno]
    yyf = yy[selector_anno]
    df = depthnp[selector_anno]

    cm = plt.get_cmap('magma')
    rndcolor = cm(1 / df / 0.15)[:, 0:3]

    selector_depth = (flow_depth_np[0, :, :] != 0) * (depthnp > 0) * (insmap_np == 0)
    flowx_depth = flow_depth_np[0][selector_depth]
    flowy_depth = flow_depth_np[1][selector_depth]

    xxf_depth = xx[selector_depth]
    yyf_depth = yy[selector_depth]
    df_depth = depthnp[selector_depth]
    rndcolor_depth = cm(1 / df_depth / 0.15)[:, 0:3]

    fig = plt.figure(figsize=(16, 9))
    fig.add_subplot(3, 1, 1)
    plt.scatter(xxf, yyf, 3, rndcolor)
    plt.imshow(image1np)

    fig.add_subplot(3, 1, 2)
    plt.scatter(xxf + flowx, yyf + flowy, 3, rndcolor)
    plt.imshow(image2np)

    fig.add_subplot(3, 1, 3)
    plt.scatter(xxf_depth + flowx_depth, yyf_depth + flowy_depth, 3, rndcolor_depth)
    plt.imshow(image2np)
    plt.show()

def depth2flow(depth, valid, intrinsic, rel_pose):
    device = depth.device
    depth = depth.squeeze().cpu().numpy()
    valid = valid.squeeze().cpu().numpy()
    intrinsic = intrinsic.squeeze().cpu().numpy()
    rel_pose = rel_pose.squeeze().cpu().numpy()
    h, w = depth.shape

    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
    selector = (valid == 1)

    xxf = xx[selector]
    yyf = yy[selector]
    df = depth[selector]

    pts3d = np.stack([xxf * df, yyf * df, df, np.ones_like(df)], axis=0)
    pts3d = np.linalg.inv(intrinsic) @ pts3d
    pts3d_oview = rel_pose @ pts3d
    pts2d_oview = intrinsic @ pts3d_oview
    pts2d_oview[0, :] = pts2d_oview[0, :] / pts2d_oview[2, :]
    pts2d_oview[1, :] = pts2d_oview[1, :] / pts2d_oview[2, :]
    selector = pts2d_oview[2, :] > 0

    flowgt = np.zeros([h, w, 2])
    flowgt[yyf.astype(np.int)[selector], xxf.astype(np.int)[selector], 0] = pts2d_oview[0, :][selector] - xxf[selector]
    flowgt[yyf.astype(np.int)[selector], xxf.astype(np.int)[selector], 1] = pts2d_oview[1, :][selector] - yyf[selector]
    flowgt = torch.from_numpy(flowgt).permute([2, 0, 1]).unsqueeze(0).cuda(device)
    return flowgt

def read_splits_mapping():
    evaluation_entries = []
    for m in range(200):
        seqname = "kittistereo15_{}/kittistereo15_{}_sync".format(str(m).zfill(6), str(m).zfill(6))
        evaluation_entries.append("{} {} {}".format(seqname, "10".zfill(10), 'l'))
    return evaluation_entries

@torch.no_grad()
def validate_poses_gpssplit(args, eval_loader, group):
    """ Peform validation using the KITTI-2015 (train) split """
    """ Peform validation using the KITTI-2015 (train) split """
    gpu = args.gpu
    eval_metrics = {'gps': torch.zeros(1).cuda(device=gpu), 'deepv2d_pose': torch.zeros(1).cuda(device=gpu),
                    'md_pose': torch.zeros(1).cuda(device=gpu), 'pixelnum': torch.zeros(1).cuda(device=gpu), 'framenum': torch.zeros(1).cuda(device=gpu)}

    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        posegt = data_blob['rel_pose'].cuda(gpu)
        if torch.sum(posegt) == 4:
            continue
        tag = data_blob['tag'][0]
        intrinsic = data_blob['intrinsic'].cuda(gpu)
        insmap = data_blob['insmap'].cuda(gpu)
        depthgt = data_blob['depthmap'].cuda(gpu)

        depthgt2 = Image.open("/media/shengjie/disk1/data/kitti_stereo15_organized/depth/{}/image_02/0000000010.png".format(tag.split(' ')[0]))
        w, h = depthgt2.size
        crph = 320
        crpw = 1216
        left = int((w - crpw) / 2)
        top = int(h - crph)
        depthgt2 = eval_loader.dataset.crop_img(np.array(depthgt2), left=left, top=top, crph=crph, crpw=crpw)
        depthgt2 = np.array(depthgt2).astype(np.float32) / 256.0
        depthgt2 = torch.from_numpy(depthgt2).unsqueeze(0).unsqueeze(0).float().cuda()

        commonareaa = (depthgt > 0) * (depthgt2 > 0)
        commonareaa = commonareaa.float()
        depthgt = depthgt * commonareaa
        depthgt2 = depthgt2 * commonareaa

        # depthgt = depthgt2
        flowgt_stereo = data_blob['flowgt_stereo'].cuda(gpu)
        valid_flow = data_blob['valid_flow'].cuda(gpu) == 1

        posepred_deepv2d = data_blob['posepred_deepv2d'].cuda(gpu)
        posepred_mD = data_blob['posepred'].cuda(gpu)[:, 0:1, :, :]

        mag = torch.sum(flowgt_stereo**2, dim=1, keepdim=True).sqrt()
        val = (mag < MAX_FLOW) * (insmap == 0) * valid_flow * (depthgt > 0)

        poses = {'gps': posegt, 'deepv2d_pose': posepred_deepv2d, 'md_pose': posepred_mD}
        for k in poses.keys():
            flow_eval = depth2flow(depthgt, depthgt > 0, intrinsic, poses[k])

            epe = torch.sum((flowgt_stereo - flow_eval)**2, dim=1, keepdim=True).sqrt()
            mag = torch.sum(flowgt_stereo**2, dim=1, keepdim=True).sqrt()

            out = ((epe[val] > 3.0) & ((epe[val]/mag[val]) > 0.05)).float()
            eval_metrics[k][0] += torch.sum(out)
        eval_metrics['pixelnum'][0] += torch.sum(val)
        eval_metrics['framenum'][0] += 1

    if args.distributed:
        for k in eval_metrics.keys():
            dist.all_reduce(tensor=eval_metrics[k], op=dist.ReduceOp.SUM, group=group)

    if args.gpu == 0:
        print_keys = ['gps', 'deepv2d_pose', 'md_pose']
        print_str = "Evaluate Poses on GPS Split with %d samples" % (eval_metrics['framenum'][0].item())
        for key in print_keys:
            print_str += ", %s-out: %f" % (key, eval_metrics[key][0].item() / eval_metrics['pixelnum'][0].item())
        print(print_str)

@torch.no_grad()
def validate_poses_fullsplit(args, eval_loader, group):
    """ Peform validation using the KITTI-2015 (train) split """
    """ Peform validation using the KITTI-2015 (train) split """
    gpu = args.gpu
    eval_metrics = {'deepv2d_pose': torch.zeros(1).cuda(device=gpu), 'pixelnum': torch.zeros(1).cuda(device=gpu), 'framenum': torch.zeros(1).cuda(device=gpu)}

    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        intrinsic = data_blob['intrinsic'].cuda(gpu)
        insmap = data_blob['insmap'].cuda(gpu)
        depthgt = data_blob['depthmap'].cuda(gpu)
        flowgt_stereo = data_blob['flowgt_stereo'].cuda(gpu)
        valid_flow = data_blob['valid_flow'].cuda(gpu) == 1

        posepred_deepv2d = data_blob['posepred_deepv2d'].cuda(gpu)

        mag = torch.sum(flowgt_stereo**2, dim=1, keepdim=True).sqrt()
        val = (mag < MAX_FLOW) * (insmap == 0) * valid_flow * (depthgt > 0)

        poses = {'deepv2d_pose': posepred_deepv2d}
        for k in poses.keys():
            flow_eval = depth2flow(depthgt, depthgt > 0, intrinsic, poses[k])

            epe = torch.sum((flowgt_stereo - flow_eval)**2, dim=1, keepdim=True).sqrt()
            mag = torch.sum(flowgt_stereo**2, dim=1, keepdim=True).sqrt()

            out = ((epe[val] > 3.0) & ((epe[val]/mag[val]) > 0.05)).float()
            eval_metrics[k][0] += torch.sum(out)
        eval_metrics['pixelnum'][0] += torch.sum(val)
        eval_metrics['framenum'][0] += 1

    if args.distributed:
        for k in eval_metrics.keys():
            dist.all_reduce(tensor=eval_metrics[k], op=dist.ReduceOp.SUM, group=group)

    if args.gpu == 0:
        print_keys = ['deepv2d_pose']
        print_str = "Evaluate Poses on Full Split %d samples" % (eval_metrics['framenum'][0].item())
        for key in print_keys:
            print_str += ", %s-out: %f" % (key, eval_metrics[key][0].item() / eval_metrics['pixelnum'][0].item())
        print(print_str)

def train(gpu, ngpus_per_node, args):
    print("Using GPU %d for training" % gpu)
    args.gpu = gpu

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=ngpus_per_node, rank=args.gpu)

    evaluation_entries = read_splits_mapping()

    eval_dataset = KITTI_eigen_stereo15(root=args.dataset_stereo15_orgned_root, inheight=args.evalheight, inwidth=args.evalwidth, entries=evaluation_entries,
                                        maxinsnum=args.maxinsnum, istrain=False, isgarg=True, deepv2dpred_root=args.deepv2dpred_root, prediction_root=args.prediction_root)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset) if args.distributed else None
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=True, num_workers=3, drop_last=False, sampler=eval_sampler)

    print("Test splits contain %d images" % (eval_dataset.__len__()))

    if args.distributed:
        group = dist.new_group([i for i in range(ngpus_per_node)])

    # validate_poses_fullsplit(args, eval_loader, group)
    validate_poses_gpssplit(args, eval_loader, group)
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

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--dataset_stereo15_orgned_root', type=str)
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