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
from exp_kitti_eigen_fixation.dataset_kitti_eigen_fixation import KITTI_eigen
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

def inf_pose_flow(rgb, sift, flow_pr_inf, insmap, mdDepth, intrinsic, pid, gradComputer=None, banins=False, samplenum=50000):
    rgbnp = rgb.squeeze().permute([1, 2, 0]).cpu().numpy().astype(np.uint8)
    grey = cv2.cvtColor(rgbnp, cv2.COLOR_BGR2GRAY)
    # Image.fromarray(grey).show()

    insmap_np = insmap[0, 0].cpu().numpy()
    intrinsicnp = intrinsic[0].cpu().numpy()
    dummyh = 370
    gradbar = 0.9
    _, _, h, w = insmap.shape
    border_sel = np.zeros([h, w])
    border_sel[int(0.25810811 * dummyh) : int(0.99189189 * dummyh)] = 1
    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')

    siftselector = np.zeros([h, w], dtype=np.bool)
    kps = sift.detect(grey, None)
    # ptsrec = list()
    for kp in kps:
        pts2dtmp = np.round(kp.pt).astype(np.int)
        if pts2dtmp[0] >= 0 and pts2dtmp[0] < w and pts2dtmp[1] >=0 and pts2dtmp[1] < h:
            siftselector[pts2dtmp[1], pts2dtmp[0]] = 1
            # ptsrec.append(pts2dtmp)
    # ptsrec_np = np.stack(ptsrec, axis=0)
    # plt.figure()
    # plt.imshow(rgbnp)
    # plt.scatter(ptsrec_np[:, 0], ptsrec_np[:, 1], 0.5)

    flow_pr_inf_x = flow_pr_inf[0, 0].cpu().numpy()
    flow_pr_inf_y = flow_pr_inf[0, 1].cpu().numpy()

    xx_nf = xx + flow_pr_inf_x
    yy_nf = yy + flow_pr_inf_y

    mdDepth_np = mdDepth.squeeze().cpu().numpy()

    if gradComputer is None:
        depth_grad = np.zeros_like(mdDepth_np)
    else:
        depth_grad = gradComputer.depth2grad(torch.from_numpy(mdDepth_np).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
        depth_grad = depth_grad / mdDepth_np

    if not banins:
        selector = (xx_nf > 0) * (xx_nf < w) * (yy_nf > 0) * (yy_nf < h) * (insmap_np == 0) * border_sel * (depth_grad < gradbar) * (siftselector == 1)
    else:
        selector = (xx_nf > 0) * (xx_nf < w) * (yy_nf > 0) * (yy_nf < h) * border_sel * (depth_grad < gradbar) * (siftselector == 1)
    selector = selector == 1

    if samplenum > np.sum(selector):
        samplenum = np.sum(selector)

    np.random.seed(pid)
    rndidx = np.random.randint(0, np.sum(selector), samplenum)

    xx_idx_sel = xx[selector][rndidx]
    yy_idx_sel = yy[selector][rndidx]

    flow_sel_mag = np.mean(np.sqrt(flow_pr_inf_x[yy_idx_sel, xx_idx_sel] ** 2 + flow_pr_inf_y[yy_idx_sel, xx_idx_sel] ** 2))

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
    scale_md = depth2scale(pts1_inliers, pts2_inliers, intrinsicnp, R, t, mdDepth_np[selector][rndidx][inliers_mask])

    if R[0, 0] < 0 or R[1, 1] < 0 or R[2, 2] < 0 or t[2] > 0:
        R = np.eye(3)
        t = np.array([[0, 0, -1]]).T
        scale_md = 0

    # Image.fromarray(flow_to_image(flow_pr_inf[0].cpu().permute([1, 2, 0]).numpy())).show()
    # tensor2disp(torch.from_numpy(selector).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()
    # tensor2disp(depthmap > 0, vmax=1, viewind=0).show()
    # tensor2disp(torch.from_numpy(selvls).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()

    return R, t, scale_md, flow_sel_mag

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
def validate_RANSAC_odom_relpose(args, eval_loader, banins=False, bangrad=False, samplenum=50000):
    if bangrad:
        gradComputer = None
    else:
        gradComputer = GradComputer()

    foldname = "banins_{}_bangrad_{}_samplenum_{}_sift".format(banins, bangrad, samplenum)
    exportfold = os.path.join(args.export_root, foldname)
    os.makedirs(exportfold, exist_ok=True)

    sift = cv2.SIFT_create()
    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        rgb = data_blob['img1']
        insmap = data_blob['insmap']
        intrinsic = data_blob['intrinsic']
        flowpred = data_blob['flowpred']
        mdDepth_pred = data_blob['mdDepth_pred']
        tag = data_blob['tag']

        R, t, scale, flow_sel_mag = inf_pose_flow(rgb, sift, flowpred, insmap, mdDepth_pred, intrinsic, int(val_id + 10), gradComputer=gradComputer, banins=banins, samplenum=samplenum)

        export_dict = {'R': R, 't': t, 'flow_sel_mag': flow_sel_mag.item(), 'scale': scale}
        frameidx = int(tag[0].split(' ')[1])
        exportfold2 = os.path.join(exportfold, tag[0].split(' ')[0], 'image_02')
        os.makedirs(exportfold2, exist_ok=True)
        export_root = os.path.join(exportfold2, str(frameidx).zfill(10) + '.pickle')
        with open(export_root, 'wb') as handle:
            pickle.dump(export_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


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

    eval_dataset = KITTI_eigen(root=args.dataset_root, inheight=args.evalheight, inwidth=args.evalwidth, entries=entries, depth_root=args.depthvlsgt_root, depthvls_root=args.depthvlsgt_root,
                               maxinsnum=args.maxinsnum, istrain=False, isgarg=True, flowPred_root=args.flowPred_root, ins_root=args.ins_root, mdPred_root=args.mdPred_root)
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=True, num_workers=args.num_workers, drop_last=False, shuffle=False)

    print("Test splits contain %d images" % (eval_dataset.__len__()))

    validate_RANSAC_odom_relpose(args, eval_loader, banins=args.banins, bangrad=args.bangrad, samplenum=args.samplenum)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--evalheight', type=int, default=1e10)
    parser.add_argument('--evalwidth', type=int, default=1e10)
    parser.add_argument('--maxinsnum', type=int, default=50)

    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--depth_root', type=str)
    parser.add_argument('--depthvlsgt_root', type=str)
    parser.add_argument('--mdPred_root', type=str)
    parser.add_argument('--flowPred_root', type=str)
    parser.add_argument('--ins_root', type=str)
    parser.add_argument('--export_root', type=str)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--banins', action='store_true')
    parser.add_argument('--bangrad', action='store_true')
    parser.add_argument('--samplenum', type=int, default=50000)


    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    train(args)