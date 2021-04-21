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
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader
from exp_kitti_eigen_fixation.dataset_kitti_eigen_fixation import KITTI_eigen, read_deepv2d_pose
from core.raft import RAFT

from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from PIL import Image, ImageDraw
from core.utils.flow_viz import flow_to_image
from core.utils.utils import InputPadder, forward_interpolate, tensor2disp, tensor2rgb, vls_ins
from posenet import Posenet
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable
from torchvision.transforms import ColorJitter

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

def depth2scale(pts2d1, pts2d2, intrinsic, R, t, coorespondedDepth):
    intrinsic33 = intrinsic[0:3, 0:3]
    M = intrinsic33 @ R @ np.linalg.inv(intrinsic33)
    delta_t = (intrinsic33 @ t).squeeze()
    minval = 1e-6


    denom = (pts2d2[0, :] * (np.expand_dims(M[2, :], axis=0) @ pts2d1).squeeze() - (np.expand_dims(M[0, :], axis=0) @ pts2d1).squeeze()) ** 2 + \
            (pts2d2[1, :] * (np.expand_dims(M[2, :], axis=0) @ pts2d1).squeeze() - (np.expand_dims(M[1, :], axis=0) @ pts2d1).squeeze()) ** 2

    selector = (denom > minval)

    rel_d = np.sqrt(
        ((delta_t[0] - pts2d2[0, selector] * delta_t[2]) ** 2 +
         (delta_t[1] - pts2d2[1, selector] * delta_t[2]) ** 2) / denom[selector])
    alpha = np.mean(coorespondedDepth[selector]) / np.mean(rel_d)
    return alpha

class GradComputer:
    def __init__(self):
        weightsx = torch.Tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([[-1., -2., -1.],
                                 [0., 0., 0.],
                                 [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)

        self.diffx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.diffy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.diffx.weight = nn.Parameter(weightsx, requires_grad=False)
        self.diffy.weight = nn.Parameter(weightsy, requires_grad=False)

    def depth2grad(self, depthmap):
        depthmap_grad = self.diffx(depthmap) ** 2 + self.diffy(depthmap) ** 2
        depthmap_grad = torch.sqrt(depthmap_grad)
        return depthmap_grad

def normalize_pts(pts):
    meanx = pts[0, :].mean()
    meany = pts[1, :].mean()
    scale = np.sqrt((pts[0, :] - meanx) ** 2 + (pts[1, :] - meany) ** 2).mean() / np.sqrt(2)

    pts_normed = np.ones_like(pts)
    pts_normed[0, :] = (pts[0, :] - meanx) / scale
    pts_normed[1, :] = (pts[1, :] - meany) / scale

    transfixM = np.eye(4)
    transfixM[0, 2] = -meanx
    transfixM[1, 2] = -meany

    scalefixM = np.eye(4)
    scalefixM[0, 0] = 1 / scale
    scalefixM[1, 1] = 1 / scale

    return pts_normed, (scalefixM @ transfixM)[0:3, 0:3]

def get_normed_ptsdist(pts2d1, pts2d2, E, intrinsic):
    intrinsic33 = intrinsic[0:3, 0:3]
    pts2d1_normed, fixM1 = normalize_pts(pts2d1)
    pts2d2_normed, fixM2 = normalize_pts(pts2d2)

    planeparam2 = np.linalg.inv(fixM2 @ intrinsic33).T @ E @ np.linalg.inv(intrinsic33) @ pts2d1
    planeparam2 = planeparam2 / np.sqrt(np.sum(planeparam2 ** 2, axis=0, keepdims=True))
    loss_dist2d_2 = np.abs(np.sum(planeparam2 * pts2d2_normed, axis=0))

    planeparam1 = (pts2d2.T @ np.linalg.inv(intrinsic33).T @ E @ np.linalg.inv(fixM1 @ intrinsic33)).T
    planeparam1 = planeparam1 / np.sqrt(np.sum(planeparam1 ** 2, axis=0, keepdims=True))
    loss_dist2d_1 = np.abs(np.sum(planeparam1 * pts2d1_normed, axis=0))

    return (loss_dist2d_1 + loss_dist2d_2) / 2

def inf_pose_flow(img1, img2, flow_pr_inf, insmap, depthmap, mdDepth, intrinsic, pid, gradComputer=None, valid_flow=None):
    insmap_np = insmap[0, 0].cpu().numpy()
    intrinsicnp = intrinsic[0].cpu().numpy()
    dummyh = 370
    samplenum = 50000
    gradbar = 0.9
    staticbar = 0.8
    _, _, h, w = img1.shape
    border_sel = np.zeros([h, w])
    border_sel[int(0.25810811 * dummyh) : int(0.99189189 * dummyh)] = 1
    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')

    flow_pr_inf_x = flow_pr_inf[0, 0].cpu().numpy()
    flow_pr_inf_y = flow_pr_inf[0, 1].cpu().numpy()

    xx_nf = xx + flow_pr_inf_x
    yy_nf = yy + flow_pr_inf_y

    depthmap_np = depthmap.squeeze().cpu().numpy()
    mdDepth_np = mdDepth.squeeze().cpu().numpy()
    if gradComputer is None:
        depth_grad = np.zeros_like(depthmap_np)
    else:
        depth_grad = gradComputer.depth2grad(torch.from_numpy(depthmap_np).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
        depth_grad = depth_grad / depthmap_np
        # tensor2disp(torch.from_numpy(depth_grad).unsqueeze(0).unsqueeze(0), percentile=95, viewind=0).show()
        # tensor2disp(torch.from_numpy(depth_grad).unsqueeze(0).unsqueeze(0) > 0.9, percentile=95, viewind=0).show()

    selector = (xx_nf > 0) * (xx_nf < w) * (yy_nf > 0) * (yy_nf < h) * (insmap_np == 0) * border_sel * (depthmap_np > 0) * (depth_grad < gradbar)
    selector = selector == 1

    if samplenum > np.sum(selector):
        samplenum = np.sum(selector)

    np.random.seed(pid)
    rndidx = np.random.randint(0, np.sum(selector), samplenum)

    xx_idx_sel = xx[selector][rndidx]
    yy_idx_sel = yy[selector][rndidx]

    if valid_flow is not None:
        valid_flow_np = valid_flow.squeeze().cpu().numpy()
        mag_sel = (xx_nf > 0) * (xx_nf < w) * (yy_nf > 0) * (yy_nf < h) * (insmap_np == 0) * border_sel * (depth_grad < gradbar) * valid_flow_np
    else:
        mag_sel = (xx_nf > 0) * (xx_nf < w) * (yy_nf > 0) * (yy_nf < h) * (insmap_np == 0) * border_sel * (depth_grad < gradbar)
    mag_sel = mag_sel == 1
    flow_sel_mag = np.mean(np.sqrt(flow_pr_inf_x[mag_sel] ** 2 + flow_pr_inf_y[mag_sel] ** 2))

    if flow_sel_mag < staticbar:
        istatic = True
    else:
        istatic = False

    # selvls = np.zeros([h, w])
    # selvls[yy_idx_sel, xx_idx_sel] = 1

    pts1 = np.stack([xx_idx_sel, yy_idx_sel], axis=1).astype(np.float)
    pts2 = np.stack([xx_nf[yy_idx_sel, xx_idx_sel], yy_nf[yy_idx_sel, xx_idx_sel]], axis=1).astype(np.float)

    E, inliers = cv2.findEssentialMat(pts1, pts2, focal=intrinsicnp[0,0], pp=(intrinsicnp[0, 2], intrinsicnp[1, 2]), method=cv2.RANSAC, prob=0.99, threshold=0.1)
    cheirality_cnt, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=intrinsicnp[0, 0], pp=(intrinsicnp[0, 2], intrinsicnp[1, 2]))

    inliers_mask = inliers == 1
    inliers_mask = np.squeeze(inliers_mask, axis=1)
    pts1_inliers = pts1[inliers_mask, :].T
    pts2_inliers = pts2[inliers_mask, :].T

    pts1_inliers = np.concatenate([pts1_inliers, np.ones([1, pts1_inliers.shape[1]])], axis=0)
    pts2_inliers = np.concatenate([pts2_inliers, np.ones([1, pts2_inliers.shape[1]])], axis=0)
    coorespondedDepth = depthmap_np[selector][rndidx][inliers_mask]
    scale = depth2scale(pts1_inliers, pts2_inliers, intrinsicnp, R, t, coorespondedDepth)
    scale_md = depth2scale(pts1_inliers, pts2_inliers, intrinsicnp, R, t, mdDepth_np[selector][rndidx][inliers_mask])

    # Image.fromarray(flow_to_image(flow_pr_inf[0].cpu().permute([1, 2, 0]).numpy())).show()
    # tensor2disp(torch.from_numpy(selector).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()
    # tensor2disp(depthmap > 0, vmax=1, viewind=0).show()
    # tensor2disp(torch.from_numpy(selvls).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()

    return R, t, scale, scale_md, flow_sel_mag, istatic

def readlines(filename):
    with open(filename, 'r') as f:
        filenames = f.readlines()
    return filenames

def rot2ang(R):
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    ang = torch.zeros([3])
    ang[0] = torch.atan2(R[2, 1], R[2, 2])
    ang[1] = torch.atan2(-R[2, 0], sy)
    ang[2] = torch.atan2(R[1, 0], R[0, 0])
    return ang

def compute_poseloss(poseest, posegt):
    testt = poseest[0:3, 3]
    testt = testt / np.sqrt(np.sum(testt ** 2))

    gtt = posegt[0:3, 3]
    gtt = gtt / np.sqrt(np.sum(gtt ** 2))

    testang = rot2ang(torch.from_numpy(poseest[0:3, 0:3]))
    gtang = rot2ang(torch.from_numpy(posegt[0:3, 0:3]))

    losst = 1 - np.sum(testt * gtt)
    lossang = np.mean(np.abs(testang.numpy() - gtang.numpy()))
    return losst, lossang

def rot2ang(R):
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    ang = torch.zeros([3])
    ang[0] = torch.atan2(R[2, 1], R[2, 2])
    ang[1] = torch.atan2(-R[2, 0], sy)
    ang[2] = torch.atan2(R[1, 0], R[0, 0])
    return ang

def compute_poseloss(poseest, posegt):
    testt = poseest[0:3, 3]
    testt = testt / torch.sqrt(torch.sum(testt ** 2))

    gtt = posegt[0:3, 3]
    gtt = gtt / torch.sqrt(torch.sum(gtt ** 2))

    testang = rot2ang(poseest[0:3, 0:3])
    gtang = rot2ang(posegt[0:3, 0:3])

    losst = 1 - torch.sum(testt * gtt)
    lossang = torch.mean(torch.abs(testang - gtang))
    return losst, lossang

@torch.no_grad()
def validate_RANSAC_odom_offline(args, seqmap, entries):
    rec_pose_scale = list()
    err_rec = {'pose_RANSAC': list(), 'pose_deepv2d': list(), 'pose_RANSAC_banins': list()}
    for val_id, entry in enumerate(tqdm(entries)):
        # Read Pose gt
        seq, frameidx, _ = entry.split(' ')
        seq = seq.split('/')[1]
        frameidx = int(frameidx)
        gtposes_sourse = readlines(os.path.join(args.odomPose_root, "{}.txt".format(str(seqmap[seq[0:21]]['mapid']).zfill(2))))
        if frameidx - int(seqmap[seq[0:21]]['stid']) < 0 or \
                frameidx + 1 - int(seqmap[seq[0:21]]['stid']) < 0 or \
                frameidx - int(seqmap[seq[0:21]]['stid']) >= len(gtposes_sourse) or \
                frameidx + 1 - int(seqmap[seq[0:21]]['stid']) >= len(gtposes_sourse):
            continue
        gtposes_str = [gtposes_sourse[frameidx - int(seqmap[seq[0:21]]['stid'])],
                       gtposes_sourse[frameidx + 1 - int(seqmap[seq[0:21]]['stid'])]]
        gtposes = list()
        for gtposestr in gtposes_str:
            gtpose = np.eye(4).flatten()
            for numstridx, numstr in enumerate(gtposestr.split(' ')):
                gtpose[numstridx] = float(numstr)
            gtpose = np.reshape(gtpose, [4, 4])
            gtposes.append(gtpose)
        posegt = np.linalg.inv(gtposes[1]) @ gtposes[0]

        pose_RANSAC_path = os.path.join(args.RANSAC_pose_root, entry.split(' ')[0], 'image_02', str(frameidx).zfill(10) + '.pickle')
        pose_RANSAC_dict = pickle.load(open(pose_RANSAC_path, "rb"))
        pose_RANSAC = np.eye(4)
        pose_RANSAC[0:3, 0:3] = pose_RANSAC_dict['R']
        pose_RANSAC[0:3, 3:4] = pose_RANSAC_dict['t']

        pose_RANSAC_banins_path = os.path.join(args.RANSAC_pose_noinsmask_root, entry.split(' ')[0], 'image_02', str(frameidx).zfill(10) + '.pickle')
        pose_RANSAC_banins_dict = pickle.load(open(pose_RANSAC_banins_path, "rb"))
        pose_RANSAC_banins = np.eye(4)
        pose_RANSAC_banins[0:3, 0:3] = pose_RANSAC_banins_dict['R']
        pose_RANSAC_banins[0:3, 3:4] = pose_RANSAC_banins_dict['t']

        pose_deepv2d_path = os.path.join(args.deepv2dpred_root, entry.split(' ')[0], 'posepred', str(frameidx).zfill(10) + '.txt')
        posepred_deepv2d = read_deepv2d_pose(pose_deepv2d_path)

        poses = dict()
        poses['pose_deepv2d'] = posepred_deepv2d
        poses['pose_RANSAC'] = pose_RANSAC
        poses['pose_RANSAC_banins'] = pose_RANSAC_banins

        for k in poses.keys():
            losst, lossang = compute_poseloss(poseest=torch.from_numpy(poses[k]), posegt=torch.from_numpy(posegt))
            err_rec[k].append([losst.item(), lossang.item()])

        rec_pose_scale.append(np.sqrt(np.sum(posegt[0:3, 3] ** 2)).item())

    rec_pose_scale = np.array(rec_pose_scale)
    binnum = 100
    bins = np.linspace(np.log(rec_pose_scale.min()), np.log(rec_pose_scale.max()), binnum)
    bins = np.exp(bins)

    indices = np.digitize(rec_pose_scale, bins[:-1])
    eval_digitized = dict()
    for k in ['pose_deepv2d', 'pose_RANSAC', 'pose_RANSAC_banins']:
        tmp_eval_metric = np.zeros([binnum, 2])
        tmp_eval_frames = np.zeros([binnum, 1])

        for idx, ii in enumerate(indices):
            tmp_eval_metric[ii] += np.log(err_rec[k][idx])
            tmp_eval_frames[ii] += 1
        tmp_eval_metric = tmp_eval_metric / (tmp_eval_frames + 1e-3)

        eval_digitized[k] = tmp_eval_metric

    fraction_scale = np.zeros([binnum, 1])
    for idx, ii in enumerate(indices):
        fraction_scale[ii] += 1
    fraction_scale = fraction_scale / np.sum(fraction_scale)

    fig = plt.figure(figsize=(16, 12))
    fig.add_subplot(3, 1, 1)
    plt.plot(bins, eval_digitized['pose_deepv2d'][:, 0])
    plt.plot(bins, eval_digitized['pose_RANSAC'][:, 0])
    plt.legend(['pose_deepv2d', 'pose_RANSAC'])
    plt.ylabel("Loss in Log")
    plt.title('Movement Direction Loss')

    fig.add_subplot(3, 1, 2)
    plt.plot(bins, eval_digitized['pose_deepv2d'][:, 1])
    plt.plot(bins, eval_digitized['pose_RANSAC'][:, 1])
    plt.legend(['pose_deepv2d', 'pose_RANSAC'])
    plt.title('Rotation Angle Loss')
    plt.ylabel("Loss in Log")

    fig.add_subplot(3, 1, 3)
    plt.plot(bins, fraction_scale)
    plt.title('Scale Percentage')
    plt.xlabel("Scale in meter")
    plt.ylabel("Percentage")
    plt.savefig(os.path.join('/home/shengjie/Desktop', 'fig1.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 12))
    fig.add_subplot(2, 1, 1)
    plt.plot(bins, eval_digitized['pose_RANSAC'][:, 0])
    plt.plot(bins, eval_digitized['pose_RANSAC_banins'][:, 0])
    plt.legend(['pose_RANSAC', 'pose_RANSAC_banins'])
    plt.ylabel("Loss in Log")
    plt.title('Movement Direction Loss')

    fig.add_subplot(2, 1, 2)
    plt.plot(bins, eval_digitized['pose_RANSAC'][:, 1])
    plt.plot(bins, eval_digitized['pose_RANSAC_banins'][:, 1])
    plt.legend(['pose_RANSAC', 'pose_RANSAC_banins'])
    plt.title('Rotation Angle Loss')
    plt.ylabel("Loss in Log")
    plt.savefig(os.path.join('/home/shengjie/Desktop', 'fig2.png'))
    plt.close()

def validate_RANSAC_odom_offline_accum(args, seqmap, entries):
    accumerr = dict()
    pos_recs = dict()
    for val_id, entry in enumerate(tqdm(entries)):
        # Read Pose gt
        seq, frameidx, _ = entry.split(' ')
        seq = seq.split('/')[1]
        if seq not in accumerr.keys():
            accumerr[seq] = {'pose_RANSAC': list(), 'pose_deepv2d': list()}
            accumpos = {'pose_RANSAC': np.array([[0, 0, 0, 1]]).T, 'pose_deepv2d': np.array([[0, 0, 0, 1]]).T,
                        'pose_gt': np.array([[0, 0, 0, 1]]).T}
            pos_recs[seq] = {'pose_RANSAC': list(), 'pose_deepv2d': list(), 'pose_gt': list()}
        frameidx = int(frameidx)
        gtposes_sourse = readlines(os.path.join(args.odomPose_root, "{}.txt".format(str(seqmap[seq[0:21]]['mapid']).zfill(2))))
        if frameidx - int(seqmap[seq[0:21]]['stid']) < 0 or \
                frameidx + 1 - int(seqmap[seq[0:21]]['stid']) < 0 or \
                frameidx - int(seqmap[seq[0:21]]['stid']) >= len(gtposes_sourse) or \
                frameidx + 1 - int(seqmap[seq[0:21]]['stid']) >= len(gtposes_sourse):
            continue
        gtposes_str = [gtposes_sourse[frameidx - int(seqmap[seq[0:21]]['stid'])],
                       gtposes_sourse[frameidx + 1 - int(seqmap[seq[0:21]]['stid'])]]
        gtposes = list()
        for gtposestr in gtposes_str:
            gtpose = np.eye(4).flatten()
            for numstridx, numstr in enumerate(gtposestr.split(' ')):
                gtpose[numstridx] = float(numstr)
            gtpose = np.reshape(gtpose, [4, 4])
            gtposes.append(gtpose)
        posegt = np.linalg.inv(gtposes[1]) @ gtposes[0]
        scalegt = np.sqrt(np.sum(posegt[0:3, 3] ** 2)).item()

        pose_RANSAC_path = os.path.join(args.RANSAC_pose_root, entry.split(' ')[0], 'image_02', str(frameidx).zfill(10) + '.pickle')
        pose_RANSAC_dict = pickle.load(open(pose_RANSAC_path, "rb"))

        pose_RANSAC = np.eye(4)
        pose_RANSAC[0:3, 0:3] = pose_RANSAC_dict['R']
        pose_RANSAC[0:3, 3:4] = pose_RANSAC_dict['t']

        if pose_RANSAC[0, 0] < 0 or pose_RANSAC[1, 1] < 0 or pose_RANSAC[2, 2] < 0 or pose_RANSAC[2, 3] > 0:
            pose_RANSAC = np.eye(4)
            pose_RANSAC[2, 3] = -1
        pose_RANSAC[0:3, 3:4] = pose_RANSAC[0:3, 3:4] * scalegt

        pose_deepv2d_path = os.path.join(args.deepv2dpred_root, entry.split(' ')[0], 'posepred', str(frameidx).zfill(10) + '.txt')
        posepred_deepv2d = read_deepv2d_pose(pose_deepv2d_path)
        # posepred_deepv2d[0:3, 3:4] = posepred_deepv2d[0:3, 3:4] / np.sqrt(np.sum(posepred_deepv2d[0:3, 3:4] ** 2)) * scalegt

        # if scalegt < 0.1:
        #     pose_RANSAC = posegt
        #     posepred_deepv2d = posegt

        poses = dict()
        poses['pose_deepv2d'] = posepred_deepv2d
        poses['pose_RANSAC'] = pose_RANSAC
        poses['pose_gt'] = posegt

        # Update Pose
        for k in accumpos.keys():
            accumpos[k] = poses[k] @ accumpos[k]
            pos_recs[seq][k].append(accumpos[k])

        for k in accumerr[seq].keys():
            loss_pos = np.sqrt(np.sum((accumpos[k] - accumpos['pose_gt']) ** 2))
            accumerr[seq][k].append(loss_pos.item())

    accumerr_fin = dict()
    print(accumerr.keys())
    for k in accumerr.keys():

        accumerr_fin[k] = dict()
        for kk in accumerr[k].keys():
            accumerr_fin[k][kk] = np.sum(np.array(accumerr[k][kk]))
            # accumerr_fin[k][kk] = np.array(accumerr[k][kk])[-1] / len(accumerr[k][kk])

        # tmppos_rec = pos_recs[k]
        #
        # plt.figure()
        # for kk in accumerr[k].keys():
        #     accum_err = np.array(accumerr[k][kk])
        #     plt.plot(accum_err)
        # plt.title("Seq %s" % k)
        # plt.show()
        #
        # plt.figure()
        # for kk in tmppos_rec.keys():
        #     tmppos = tmppos_rec[kk]
        #     tmppos = np.stack(tmppos, axis=0)
        #     plt.plot(tmppos[0:1000, 0, 0], tmppos[0:1000, 2, 0])
        # plt.legend(list(tmppos_rec.keys()))
        # plt.savefig('/home/shengjie/Desktop/fig3.png')
        #
        # return
    print(accumerr_fin)

def validate_RANSAC_odom_offline_accum_absl(args, seqmap, entries):
    # entries = entries[0:1000]
    accumerr = dict()
    pos_recs = dict()
    for val_id, entry in enumerate(tqdm(entries)):
        # Read Pose gt
        seq, frameidx, _ = entry.split(' ')
        seq = seq.split('/')[1]
        if seq not in accumerr.keys():
            accumerr[seq] = {'pose_RANSAC': list(), 'pose_deepv2d': list()}
            accumpos = {'pose_RANSAC': np.array([[0, 0, 0, 1]]).T, 'pose_deepv2d': np.array([[0, 0, 0, 1]]).T,
                        'pose_gt': np.array([[0, 0, 0, 1]]).T}
            pos_recs[seq] = {'pose_RANSAC': list(), 'pose_deepv2d': list(), 'pose_gt': list()}
        frameidx = int(frameidx)
        gtposes_sourse = readlines(os.path.join(args.odomPose_root, "{}.txt".format(str(seqmap[seq[0:21]]['mapid']).zfill(2))))
        if frameidx - int(seqmap[seq[0:21]]['stid']) < 0 or \
                frameidx + 1 - int(seqmap[seq[0:21]]['stid']) < 0 or \
                frameidx - int(seqmap[seq[0:21]]['stid']) >= len(gtposes_sourse) or \
                frameidx + 1 - int(seqmap[seq[0:21]]['stid']) >= len(gtposes_sourse):
            continue
        gtposes_str = [gtposes_sourse[frameidx - int(seqmap[seq[0:21]]['stid'])],
                       gtposes_sourse[frameidx + 1 - int(seqmap[seq[0:21]]['stid'])]]
        gtposes = list()
        for gtposestr in gtposes_str:
            gtpose = np.eye(4).flatten()
            for numstridx, numstr in enumerate(gtposestr.split(' ')):
                gtpose[numstridx] = float(numstr)
            gtpose = np.reshape(gtpose, [4, 4])
            gtposes.append(gtpose)
        posegt = np.linalg.inv(gtposes[1]) @ gtposes[0]
        scalegt = np.sqrt(np.sum(posegt[0:3, 3] ** 2)).item()

        pose_deepv2d_path = os.path.join(args.deepv2dpred_root, entry.split(' ')[0], 'posepred', str(frameidx).zfill(10) + '.txt')
        posepred_deepv2d = read_deepv2d_pose(pose_deepv2d_path)

        pose_RANSAC_path = os.path.join(args.RANSAC_pose_root, entry.split(' ')[0], 'image_02', str(frameidx).zfill(10) + '.pickle')
        pose_RANSAC_dict = pickle.load(open(pose_RANSAC_path, "rb"))

        pose_RANSAC = np.eye(4)
        pose_RANSAC[0:3, 0:3] = pose_RANSAC_dict['R']
        pose_RANSAC[0:3, 3:4] = pose_RANSAC_dict['t']

        if pose_RANSAC[0, 0] < 0 or pose_RANSAC[1, 1] < 0 or pose_RANSAC[2, 2] < 0 or pose_RANSAC[2, 3] > 0:
            pose_RANSAC = np.eye(4)
            pose_RANSAC[2, 3] = -1
        # pose_RANSAC[0:3, 3:4] = pose_RANSAC[0:3, 3:4] * pose_RANSAC_dict['scale_md']
        # pose_RANSAC[0:3, 3:4] = pose_RANSAC[0:3, 3:4] * pose_RANSAC_dict['scale']
        # pose_RANSAC[0:3, 3:4] = pose_RANSAC[0:3, 3:4] * pose_RANSAC_dict['scale_md_2']
        # pose_RANSAC[0:3, 3:4] = pose_RANSAC[0:3, 3:4] * np.sqrt(np.sum(posepred_deepv2d[0:3, 3:4] ** 2))
        pose_RANSAC[0:3, 3:4] = pose_RANSAC[0:3, 3:4] * scalegt

        poses = dict()
        poses['pose_deepv2d'] = posepred_deepv2d
        poses['pose_RANSAC'] = pose_RANSAC
        poses['pose_gt'] = posegt

        # Update Pose
        for k in accumpos.keys():
            accumpos[k] = poses[k] @ accumpos[k]
            pos_recs[seq][k].append(accumpos[k])

        for k in accumerr[seq].keys():
            loss_pos = np.sqrt(np.sum((accumpos[k] - accumpos['pose_gt']) ** 2))
            accumerr[seq][k].append(loss_pos.item())

    # accumerr_fin = dict()
    # for k in accumerr.keys():
    #     accumerr_fin[k] = dict()
    #     for kk in accumerr[k].keys():
    #         accumerr_fin[k][kk] = np.array(accumerr[k][kk])[-1] / len(accumerr[k][kk])
    # print(accumerr_fin)

    accumerr_fin = dict()
    for k in accumerr.keys():
        accumerr_fin[k] = dict()
        for kk in accumerr[k].keys():
            # accumerr_fin[k][kk] = np.sum(np.array(accumerr[k][kk]))
            accumerr_fin[k][kk] = np.array(accumerr[k][kk])[-1] / len(accumerr[k][kk])

        if k == '2011_09_30_drive_0016_sync':
            tmppos_rec = pos_recs[k]

            # plt.figure()
            # for kk in accumerr[k].keys():
            #     accum_err = np.array(accumerr[k][kk])
            #     plt.plot(accum_err)
            # plt.title("Seq %s" % k)
            # plt.legend(accumerr[k].keys())
            # plt.show()

            # plt.figure()
            # for kk in tmppos_rec.keys():
            #     tmppos = tmppos_rec[kk]
            #     tmppos = np.stack(tmppos, axis=0)
            #     plt.plot(tmppos[:,0,0], tmppos[:,2,0])
            # plt.legend(list(tmppos_rec.keys()))
            # plt.axis('scaled')
            # plt.show()
            # plt.savefig('/home/shengjie/Desktop/fig3.png')
            #
            # return
    print(accumerr_fin)

def generate_seqmapping():
    seqmapping = \
    ['00 2011_10_03_drive_0027 000000 004540',
     "04 2011_09_30_drive_0016 000000 000270",
     "05 2011_09_30_drive_0018 000000 002760",
     "07 2011_09_30_drive_0027 000000 001100"]

    entries = list()
    seqmap = dict()
    for seqm in seqmapping:
        mapentry = dict()
        mapid, seqname, stid, enid = seqm.split(' ')
        mapentry['mapid'] = int(mapid)
        mapentry['stid'] = int(stid)
        mapentry['enid'] = int(enid)
        seqmap[seqname] = mapentry

        for k in range(int(stid), int(enid)):
            entries.append("{}/{}_sync {} {}".format(seqname[0:10], seqname, str(k).zfill(10), 'l'))

    return seqmap, entries

def train(args):
    seqmap, entries = generate_seqmapping()

    entries = entries
    # validate_RANSAC_odom_offline(args, seqmap, entries)
    # validate_RANSAC_odom_offline_accum(args, seqmap, entries)
    validate_RANSAC_odom_offline_accum_absl(args, seqmap, entries)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")

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

    parser.add_argument('--RANSAC_pose_root', type=str)
    parser.add_argument('--RANSAC_pose_noinsmask_root', type=str)
    parser.add_argument('--deepv2dpred_root', type=str)
    parser.add_argument('--odomPose_root', type=str)
    parser.add_argument('--num_workers', type=int, default=12)


    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    torch.cuda.empty_cache()

    train(args)