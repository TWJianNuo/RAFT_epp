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
from exp_kitti_eigen_fixation.inference.dataset_gen_ours_only import KITTI_eigen
from exp_kitti_eigen_fixation.eppflowenet.EppFlowNet_scale_initialD import EppFlowNet

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

def scale_invariant(gt, pr):
    """
    Computes the scale invariant loss based on differences of logs of depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)
    depth1:  one depth map
    depth2:  another depth map
    Returns:
        scale_invariant_distance
    """
    gt = gt.reshape(-1)
    pr = pr.reshape(-1)

    v = gt > 0.1
    gt = gt[v]
    pr = pr[v]

    log_diff = np.log(gt) - np.log(pr)
    num_pixels = np.float32(log_diff.size)

    # sqrt(Eq. 3)
    return np.sqrt(np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(num_pixels))

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

    sc_inv = scale_invariant(gt, pred)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

@torch.no_grad()
def validate_kitti(model, args, eval_loader):
    """ Peform validation using the KITTI-2015 (train) split """
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    gpu = args.gpu
    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        image1 = data_blob['img1'].cuda(gpu) / 255.0
        image2 = data_blob['img2'].cuda(gpu) / 255.0
        intrinsic = data_blob['intrinsic'].cuda(gpu)
        insmap = data_blob['insmap'].cuda(gpu)
        posepred = data_blob['posepred'].cuda(gpu)
        tag = data_blob['tag'][0]
        seq, frmidx = tag.split(' ')
        size_str = data_blob['size_str'][0]
        orgh, orgw = size_str.split('_')
        orgh = int(orgh)
        orgw = int(orgw)

        left = int((orgw - args.evalwidth) / 2)
        top = int(orgh - args.evalheight)

        predread_np_orgsize = np.zeros([orgh, orgw])

        mD_pred = data_blob['mdDepth_pred'].cuda(gpu)

        mD_pred_clipped = torch.clamp_min(mD_pred, min=args.min_depth_pred)

        outputs = model(image1, image2, mD_pred_clipped, intrinsic, posepred, insmap)
        predread = outputs[('depth', 2)]
        predread_np = predread.squeeze().cpu().numpy()
        predread_np_orgsize[top:top+args.evalheight, left:left+args.evalwidth] = predread_np

        predread_np_orgsize = (predread_np_orgsize * 256.0).astype(np.uint16)

        fold_root = os.path.join(args.export_root, seq, 'image_02')
        img_root = os.path.join(fold_root, '{}.png'.format(frmidx))
        os.makedirs(fold_root, exist_ok=True)
        Image.fromarray(predread_np_orgsize).save(img_root)

MAX_FLOW = 400

def get_odomentries(args):
    import glob
    odomentries = list()
    odomseqs = [
        '2011_10_03/2011_10_03_drive_0027_sync',
        '2011_09_30/2011_09_30_drive_0018_sync',
    ]
    for odomseq in odomseqs:
        leftimgs = glob.glob(os.path.join(args.odom_root, odomseq, 'image_02/data', "*.png"))
        for leftimg in leftimgs:
            imgname = os.path.basename(leftimg)
            odomentries.append("{} {} {}".format(odomseq, imgname.rstrip('.png'), 'l'))
    return odomentries

def read_splits(args):
    split_root = os.path.join(project_rootdir, 'exp_pose_mdepth_kitti_eigen/splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'train_files.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'test_files.txt'), 'r')]
    odom_entries = get_odomentries(args)
    return evaluation_entries + odom_entries + train_entries
    # return evaluation_entries

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

    evaluation_entries = read_splits(args)

    eval_dataset = KITTI_eigen(root=args.dataset_root, inheight=args.evalheight, inwidth=args.evalwidth, entries=evaluation_entries, maxinsnum=args.maxinsnum,
                               depth_root=args.depth_root, depthvls_root=args.depthvlsgt_root, mdPred_root=args.mdPred_root,
                               ins_root=args.ins_root, istrain=False, isgarg=True, RANSACPose_root=args.RANSACPose_root, baninsmap=args.baninsmap, bandepth=True)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset) if args.distributed else None
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=True, num_workers=3, drop_last=False, sampler=eval_sampler)

    print("Test splits contain %d images" % (eval_dataset.__len__()))

    validate_kitti(model.module, args, eval_loader)
    return

def validate_kitti_gen():
    split_root = os.path.join(project_rootdir, 'exp_pose_mdepth_kitti_eigen/splits')
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'test_files.txt'), 'r')]
    eval_measures_depth_nps = list()
    totcount = 0
    for val_id, entry in enumerate(tqdm(evaluation_entries)):
        seq, frmidx, _ = entry.split(' ')
        depthgt_path = os.path.join(args.depth_root, seq, "image_02/{}.png".format(str(frmidx).zfill(10)))
        depthpred_path = os.path.join(args.export_root, seq, "image_02/{}.png".format(str(frmidx).zfill(10)))

        if not os.path.exists(depthgt_path):
            continue
        totcount += 1

        depthgt = np.array(Image.open(depthgt_path)).astype(np.float32) / 256.0
        depthpred = np.array(Image.open(depthpred_path)).astype(np.float32) / 256.0

        orgh, orgw = depthgt.shape

        crop = np.array([0.40810811 * orgh, 0.99189189 * orgh, 0.03594771 * orgw, 0.96405229 * orgw]).astype(np.int32)
        crop_mask = np.zeros([orgh, orgw])
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        depthgt = depthgt * crop_mask

        selector = ((depthgt > 0) * (depthpred > 0) * (depthgt > args.min_depth_eval) * (depthgt < args.max_depth_eval)).astype(np.float32)
        depthpred = np.clip(depthpred, a_min=args.min_depth_eval, a_max=args.max_depth_eval)
        depth_gt_flatten = depthgt[selector == 1]
        pred_depth_flatten = depthpred[selector == 1]

        pred_depth_flatten = np.median(depth_gt_flatten/pred_depth_flatten) * pred_depth_flatten

        eval_measures_depth_np = compute_errors(gt=depth_gt_flatten, pred=pred_depth_flatten)
        eval_measures_depth_nps.append(eval_measures_depth_np)
    eval_measures_depth_nps = np.array(eval_measures_depth_nps)
    eval_measures_depth_nps = np.mean(eval_measures_depth_nps, axis=0)
    print('Computing Depth errors for %f eval samples' % (totcount))
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    for i in range(8):
        print('{:7.3f}, '.format(eval_measures_depth_nps[i]), end='')
    print('{:7.3f}'.format(eval_measures_depth_nps[8]))



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
    parser.add_argument('--export_root', type=str)
    parser.add_argument('--odom_root', type=str)

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

    validate_kitti_gen()
