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
def validate_kitti(model, args, eval_loader, logger, group, total_steps, isdeepv2d=False):
    """ Peform validation using the KITTI-2015 (train) split """
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    gpu = args.gpu
    eval_measures_depth = torch.zeros(10).cuda(device=gpu)
    err_rec = list()
    err_rec_deepv2d = list()
    err_rec_md = list()
    mv_rec = list()
    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        image1 = data_blob['img1'].cuda(gpu) / 255.0
        image2 = data_blob['img2'].cuda(gpu) / 255.0
        intrinsic = data_blob['intrinsic'].cuda(gpu)
        insmap = data_blob['insmap'].cuda(gpu)
        posepred = data_blob['posepred'].cuda(gpu)
        depthgt = data_blob['depthmap'].cuda(gpu)

        rel_pose = data_blob['rel_pose'][0].cpu().numpy()
        gps_scale = np.sqrt(np.sum(rel_pose[0:3, 3] ** 2))

        if not args.initbymD:
            mD_pred = data_blob['depthpred'].cuda(gpu)
        else:
            mD_pred = data_blob['mdDepth_pred'].cuda(gpu)

        mD_pred_clipped = torch.clamp_min(mD_pred, min=args.min_depth_pred)

        if not isdeepv2d:
            outputs = model(image1, image2, mD_pred_clipped, intrinsic, posepred, insmap)
            predread = outputs[('depth', 2)]
        else:
            depthpred_deepv2d = data_blob['depthpred_deepv2d'].cuda(gpu)
            predread = depthpred_deepv2d
            # predread = data_blob['mdDepth_pred'].cuda(gpu)

        selector = ((depthgt > 0) * (predread > 0) * (depthgt > args.min_depth_eval) * (depthgt < args.max_depth_eval)).float()
        predread = torch.clamp(predread, min=args.min_depth_eval, max=args.max_depth_eval)
        depth_gt_flatten = depthgt[selector == 1].cpu().numpy()
        pred_depth_flatten = predread[selector == 1].cpu().numpy()
        deepv2d_depth_flatten = data_blob['depthpred_deepv2d'][selector == 1].cpu().numpy()
        mD_pred_clipped_flatten = mD_pred[selector == 1].cpu().numpy()

        eval_measures_depth_np = compute_errors(gt=depth_gt_flatten, pred=pred_depth_flatten)
        eval_measures_depth_deepv2d_np = compute_errors(gt=depth_gt_flatten, pred=deepv2d_depth_flatten)
        eval_measures_depth_md_np = compute_errors(gt=depth_gt_flatten, pred=mD_pred_clipped_flatten)

        err_rec.append(eval_measures_depth_np[-3])
        mv_rec.append(gps_scale)
        err_rec_deepv2d.append(eval_measures_depth_deepv2d_np[-3])
        err_rec_md.append(eval_measures_depth_md_np[-3])

    err_rec = np.array(err_rec)
    mv_rec = np.array(mv_rec)
    err_rec_deepv2d = np.array(err_rec_deepv2d)
    err_rec_md = np.array(err_rec_md)

    check_dist = np.linspace(0, 3, 200)
    dist = 0.4

    plot_mv = list()
    plot_err = list()
    plot_std = list()
    for d in check_dist:
        d_low = d * (1 - dist)
        d_hig = d * (1 + dist)

        selector = (mv_rec >= d_low) * (mv_rec <= d_hig)
        if np.sum(selector) < 5:
            continue
        else:
            err1 = np.mean(err_rec[selector])
            err2 = np.mean(err_rec_deepv2d[selector])
            err3 = np.mean(err_rec_md[selector])

            std1 = np.std(err_rec[selector])
            std2 = np.std(err_rec_deepv2d[selector])
            std3 = np.std(err_rec_md[selector])
            plot_err.append(np.array([err1, err2, err3]))
            plot_std.append(np.array([std1, std2, std3]))
            plot_mv.append(d)

    plot_err = np.stack(plot_err, axis=0)
    plot_mv = np.array(plot_mv)
    plot_std = np.stack(plot_std, axis=0)

    thickness = 0.1

    fig, ax = plt.subplots()
    plt.plot(plot_mv, plot_err[:, 0])
    plt.plot(plot_mv, plot_err[:, 1])
    plt.plot(plot_mv, plot_err[:, 2])
    ax.fill_between(plot_mv, plot_err[:, 0] - plot_std[:, 0] * thickness, plot_err[:, 0] + plot_std[:, 0] * thickness, alpha=0.5)
    ax.fill_between(plot_mv, plot_err[:, 1] - plot_std[:, 1] * thickness, plot_err[:, 1] + plot_std[:, 1] * thickness, alpha=0.5)
    ax.fill_between(plot_mv, plot_err[:, 2] - plot_std[:, 2] * thickness, plot_err[:, 2] + plot_std[:, 2] * thickness, alpha=0.5)
    plt.xlabel('scale in meters')
    plt.ylabel('a1')
    plt.legend(['ours', 'DeepV2D', 'Bts'], bbox_to_anchor=(0.1, 0.3))
    plt.title("Error curve in KITTI")
    plt.savefig('/home/shengjie/Desktop/1.png', bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    plt.show()






    ave_err_rec = np.zeros((bins.shape[0], 3))
    ave_err_rec_count = np.zeros((bins.shape[0], 1))
    for idx, indice in enumerate(indices):
        ave_err_rec[indice, 0] += err_rec[idx]
        ave_err_rec[indice, 1] += err_rec_deepv2d[idx]
        ave_err_rec[indice, 2] += err_rec_md[idx]
        ave_err_rec_count[indice, 0] += 1
    ave_err_rec = ave_err_rec / (ave_err_rec_count + 1e-6)
    plt.figure()
    plt.plot(bins, ave_err_rec[:, 0])
    plt.plot(bins, ave_err_rec[:, 1])
    plt.show()

    plt.figure()
    plt.scatter(mv_rec, err_rec)
    plt.scatter(mv_rec, err_rec_deepv2d)
    plt.show()


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

    logroot = os.path.join(args.logroot, args.name)
    print("Parameter Count: %d, saving location: %s" % (count_parameters(model), logroot))

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
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=True, num_workers=3, drop_last=False, sampler=eval_sampler)

    print("Test splits contain %d images" % (eval_dataset.__len__()))

    if args.distributed:
        group = dist.new_group([i for i in range(ngpus_per_node)])

    validate_kitti(model.module, args, eval_loader, None, group, None, isdeepv2d=False)
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