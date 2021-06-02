from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")

import os, sys
project_rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import pickle
from core.utils.frame_utils import readFlowKITTI

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader
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
from core.utils import frame_utils
import copy
from tqdm import tqdm
import glob
import time

import matplotlib.pyplot as plt

def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data

def get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam):
    pose_imu2cam = np.eye(4)
    pose_imu2cam[0:3, 0:3] = np.reshape(imu2cam['R'], [3, 3])
    pose_imu2cam[0:3, 3] = imu2cam['T']

    pose_velo2cam = np.eye(4)
    pose_velo2cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
    pose_velo2cam[0:3, 3] = velo2cam['T']

    R_rect_00 = np.eye(4)
    R_rect_00[0:3, 0:3] = cam2cam['R_rect_00'].reshape(3, 3)

    intrinsic = np.eye(4)
    intrinsic[0:3, 0:3] = cam2cam['P_rect_02'].reshape(3, 4)[0:3, 0:3]

    org_intrinsic = np.eye(4)
    org_intrinsic[0:3, :] = cam2cam['P_rect_02'].reshape(3, 4)
    extrinsic_from_intrinsic = np.linalg.inv(intrinsic) @ org_intrinsic
    extrinsic_from_intrinsic[0:3, 0:3] = np.eye(3)

    extrinsic = extrinsic_from_intrinsic @ R_rect_00 @ pose_velo2cam @ pose_imu2cam

    return intrinsic.astype(np.float32), extrinsic.astype(np.float32)

class NYUV2(data.Dataset):
    def __init__(self, root, entries, iter, flowPred_root=None, mdPred_root=None):
        super(NYUV2, self).__init__()

        self.mdPred_root = mdPred_root
        self.flowPred_root = flowPred_root
        self.root = root

        self.image_list = list()
        self.depthgt_list = list()
        self.entries = list()
        self.flowPred_list = list()
        self.mdPred_list = list()

        fx_rgb = 5.1885790117450188e+02
        fy_rgb = 5.1946961112127485e+02
        cx_rgb = 3.2558244941119034e+02
        cy_rgb = 2.5373616633400465e+02
        self.intrinsic = np.eye(4)
        self.intrinsic[0, 0] = fx_rgb
        self.intrinsic[1, 1] = fy_rgb
        self.intrinsic[0, 2] = cx_rgb
        self.intrinsic[1, 2] = cy_rgb

        for entry in entries:
            seq, index = entry.split(' ')
            index = int(index)

            img1path = os.path.join(self.root, seq, 'rgb_{}.png'.format(str(index).zfill(5)))
            img2path = os.path.join(self.root, seq, 'rgb_{}.png'.format(str(index + 1).zfill(5)))

            if not os.path.exists(img1path):
                img1path = img1path.replace('.png', '.jpg')
                img2path = img2path.replace('.png', '.jpg')

            depthpath = os.path.join(self.root, seq, 'sync_depth_{}.png'.format(str(index).zfill(5)))

            mdDepth_path = os.path.join(self.mdPred_root, str(iter).zfill(3), seq, "sync_depth_{}.png".format(str(index).zfill(5)))
            flowpred_path = os.path.join(self.flowPred_root, seq, "{}.png".format(str(index).zfill(5)))

            if not os.path.exists(mdDepth_path):
                raise Exception("mD pred file %s missing" % mdDepth_path)

            if not os.path.exists(img2path):
                self.image_list.append([img1path, img1path])
            else:
                self.image_list.append([img1path, img2path])

            self.entries.append(entry)
            self.flowPred_list.append(flowpred_path)
            self.mdPred_list.append(mdDepth_path)
            self.depthgt_list.append(depthpath)

        assert len(self.entries) == len(self.image_list)

    def __getitem__(self, index):
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        intrinsic = copy.deepcopy(self.intrinsic)

        mdDepth = cv2.imread(self.mdPred_list[index], -1)
        mdDepth = mdDepth.astype(np.float32) / 1000.0

        gtDepth = cv2.imread(self.depthgt_list[index], -1)
        gtDepth = gtDepth.astype(np.float32) / 1000.0

        flowpred_RAFT, valid_flow = readFlowKITTI(self.flowPred_list[index])
        data_blob = self.wrapup(img1=img1, img2=img2, intrinsic=intrinsic, mdDepth=mdDepth, gtDepth=gtDepth, flowpred_RAFT=flowpred_RAFT, tag=self.entries[index])
        return data_blob

    def wrapup(self, img1, img2, intrinsic, mdDepth, gtDepth, flowpred_RAFT, tag):
        img1 = torch.from_numpy(img1).permute([2, 0, 1]).float()
        img2 = torch.from_numpy(img2).permute([2, 0, 1]).float()
        intrinsic = torch.from_numpy(intrinsic).float()
        mdDepth = torch.from_numpy(mdDepth).unsqueeze(0)
        gtDepth = torch.from_numpy(gtDepth).unsqueeze(0)
        flowpred_RAFT = torch.from_numpy(flowpred_RAFT).permute([2, 0, 1]).float()

        data_blob = dict()
        data_blob['img1'] = img1
        data_blob['img2'] = img2
        data_blob['intrinsic'] = intrinsic
        data_blob['mdDepth'] = mdDepth
        data_blob['gtDepth'] = gtDepth
        data_blob['flowpred_RAFT'] = flowpred_RAFT
        data_blob['tag'] = tag

        return data_blob

    def __len__(self):
        return len(self.entries)


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

def inf_pose_flow(flow_pr_inf, mdDepth, intrinsic, pid, gradComputer=None, samplenum=50000, ban_scale_check=False):
    intrinsicnp = intrinsic[0].cpu().numpy()
    gradbar = 0.6
    _, _, h, w = mdDepth.shape
    border_sel = np.zeros([h, w])
    border_sel[45:471, 41:601] = 1
    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')

    flow_pr_inf_x = flow_pr_inf[0, 0].cpu().numpy()
    flow_pr_inf_y = flow_pr_inf[0, 1].cpu().numpy()

    xx_nf = xx + flow_pr_inf_x
    yy_nf = yy + flow_pr_inf_y

    mdDepth_np = mdDepth.squeeze().cpu().numpy()

    depth_grad = gradComputer.depth2grad(torch.from_numpy(mdDepth_np).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
    depth_grad = depth_grad / mdDepth_np
    # tensor2disp(torch.from_numpy(depth_grad).unsqueeze(0).unsqueeze(0), vmax=gradbar, viewind=0).show()

    selector = (xx_nf > 0) * (xx_nf < w) * (yy_nf > 0) * (yy_nf < h) * border_sel * (depth_grad < gradbar) * (mdDepth_np > 0)
    selector = selector == 1

    # tensor2disp(1 / mdDepth, vmax=1, viewind=0).show()
    # tensor2disp(torch.from_numpy(selector).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()

    if samplenum > np.sum(selector):
        samplenum = np.sum(selector)

    np.random.seed(pid)
    rndidx = np.random.randint(0, np.sum(selector), samplenum)

    xx_idx_sel = xx[selector][rndidx]
    yy_idx_sel = yy[selector][rndidx]

    # flow_sel_mag = np.mean(np.sqrt(flow_pr_inf_x[yy_idx_sel, xx_idx_sel] ** 2 + flow_pr_inf_y[yy_idx_sel, xx_idx_sel] ** 2))

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

    if R[0, 0] < 0 or R[1, 1] < 0 or R[2, 2] < 0 or pts1_inliers.shape[1] <= 2000:
        R = np.eye(3)
        t = np.array([[0, 0, -1]]).T
        scale_md = 0
        bestid = -2
    else:
        if ban_scale_check:
            scale_md = scale_md
            bestid = -2
        else:
            scale_md, bestid = select_scale(scale_md, R, t, pts1_inliers, pts2_inliers, mdDepth_np[pts1_inliers[1, :].astype(np.int), pts1_inliers[0, :].astype(np.int)], intrinsicnp)

    # Image.fromarray(flow_to_image(flow_pr_inf[0].cpu().permute([1, 2, 0]).numpy())).show()
    # tensor2disp(torch.from_numpy(selector).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()
    # tensor2disp(depthmap > 0, vmax=1, viewind=0).show()
    # tensor2disp(torch.from_numpy(selvls).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()

    return R, t, scale_md, pts1_inliers, pts2_inliers, bestid

def select_scale(scale_md, R, t, pts1_inliers, pts2_inliers, mdDepth_npf, intrinsicnp):
    numres = 49

    divrat = 5
    maxrat = 1
    pos = (np.exp(np.linspace(0, divrat, numres)) - 1) / (np.exp(divrat) - 1) * np.exp(maxrat) + 1
    neg = np.exp(-np.log(pos))
    tot = np.sort(np.concatenate([pos, neg, np.array([1e-5])]))

    scale_md_cand = scale_md * tot

    self_pose = np.eye(4)
    self_pose[0:3, 0:3] = R
    self_pose[0:3, 3:4] = t
    self_pose = np.expand_dims(self_pose, axis=0)
    self_pose = np.repeat(self_pose, axis=0, repeats=tot.shape[0])
    self_pose[:, 0:3, 3] = self_pose[:, 0:3, 3] * np.expand_dims(scale_md_cand, axis=1)

    pts3d = np.stack([pts1_inliers[0, :] * mdDepth_npf, pts1_inliers[1, :] * mdDepth_npf, mdDepth_npf, np.ones_like(mdDepth_npf)])
    pts3d = intrinsicnp @ self_pose @ np.linalg.inv(intrinsicnp) @ np.repeat(np.expand_dims(pts3d, axis=0), axis=0, repeats=tot.shape[0])
    pts3d[:, 0, :] = pts3d[:, 0, :] / pts3d[:, 2, :]
    pts3d[:, 1, :] = pts3d[:, 1, :] / pts3d[:, 2, :]

    loss = np.mean(np.abs(pts3d[:, 0, :] - pts2_inliers[0]), axis=1) + np.mean(np.abs(pts3d[:, 1, :] - pts2_inliers[1]), axis=1)
    np.abs(pts1_inliers[0] - pts2_inliers[0]).mean() + np.abs(pts1_inliers[1] - pts2_inliers[1]).mean()

    best = np.argmin(loss)
    # plt.figure()
    # plt.plot(loss)
    # plt.show()
    return scale_md_cand[best], best

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
def validate_RANSAC_odom_relpose(args, eval_loader, samplenum=50000, iters=0):
    gradComputer = GradComputer()
    # optimized = 0
    # skipped = 0
    # unresolved = 0
    bestids = list()
    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        intrinsic = data_blob['intrinsic']
        flowpred = data_blob['flowpred_RAFT']
        mdDepth_pred = data_blob['mdDepth']
        # mdDepth_pred = data_blob['gtDepth']
        tag = data_blob['tag'][0]

        seq, frmidx = tag.split(' ')
        exportfold = os.path.join(args.export_root, str(iters).zfill(3), seq)
        os.makedirs(exportfold, exist_ok=True)
        export_root = os.path.join(exportfold, frmidx.zfill(5) + '.pickle')

        if torch.sum(torch.abs(data_blob['img1'] - data_blob['img2'])) < 1:
            R = np.eye(3)
            t = np.array([[0, 0, -1]]).T
            scale = 1e-10
            bestid = -1
        else:
            R, t, scale, pts1_inliers, pts2_inliers, bestid = inf_pose_flow(flowpred, mdDepth_pred, intrinsic, int(iters * eval_loader.__len__() + val_id), gradComputer=gradComputer, samplenum=samplenum, ban_scale_check=args.ban_scale_check)

        self_pose = np.eye(4)
        self_pose[0:3, 0:3] = R
        self_pose[0:3, 3:4] = t * scale
        bestids.append(bestid)

        with open(export_root, 'wb') as handle:
            pickle.dump(self_pose, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # if bestid >= 0:
        #     optimized += 1
        # elif bestid == -1:
        #     skipped += 1
        # else:
        #     unresolved += 1
        #
        # if bestid == -1:
        #     flow_numpy = flowpred[0].cpu().permute([1, 2, 0]).numpy()
        #     _, _, h, w = mdDepth_pred.shape
        #     xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        #     xxf = xx.flatten()
        #     yyf = yy.flatten()
        #     rndidx = np.random.randint(0, xxf.shape[0], 100)
        #     xxf_rnd = xxf[rndidx]
        #     yyf_rnd = yyf[rndidx]
        #
        #     selector = mdDepth_pred.squeeze().cpu().numpy()[yyf_rnd, xxf_rnd] > 0
        #     xxf_rnd = xxf_rnd[selector]
        #     yyf_rnd = yyf_rnd[selector]
        #
        #     flowx = flow_numpy[yyf_rnd, xxf_rnd, 0]
        #     flowy = flow_numpy[yyf_rnd, xxf_rnd, 1]
        #     depthf = mdDepth_pred.squeeze().cpu().numpy()[yyf_rnd, xxf_rnd]
        #
        #     intrinsic_np = intrinsic.squeeze().numpy()
        #     pts3d = np.stack([xxf_rnd * depthf, yyf_rnd * depthf, np.ones_like(yyf_rnd) * depthf, np.ones_like(yyf_rnd)], axis=0)
        #     pts3d = intrinsic_np @ self_pose @ np.linalg.inv(intrinsic_np) @ pts3d
        #     pts3d[0, :] = pts3d[0, :] / pts3d[2, :]
        #     pts3d[1, :] = pts3d[1, :] / pts3d[2, :]
        #
        #     image1 = data_blob['img1']
        #     image2 = data_blob['img2']
        #     vlsrgb1 = tensor2rgb(image1 / 255.0, viewind=0)
        #     vlsrgb2 = tensor2rgb(image2 / 255.0, viewind=0)
        #
        #     fig = plt.figure(figsize=(12, 9))
        #     plt.subplot(2, 1, 1)
        #     plt.scatter(xxf_rnd, yyf_rnd, 1, 'r')
        #     plt.imshow(vlsrgb1)
        #     plt.subplot(2, 1, 2)
        #     plt.scatter(flowx + xxf_rnd, flowy + yyf_rnd, 1, 'r')
        #     plt.scatter(pts3d[0, :], pts3d[1, :], 1, 'b')
        #     plt.imshow(vlsrgb2)
        #     plt.savefig(os.path.join("/media/shengjie/disk1/visualization/nyu_pose_vls", "{}_{}.jpg".format(seq, str(frmidx).zfill(5))))
        #     plt.close()

    # print("Optimized: %f, unresolved: %f, skipped: %f" % (optimized / (optimized +skipped + unresolved), unresolved / (optimized +skipped + unresolved), skipped / (optimized +skipped + unresolved)))
    # bestids = np.array(bestids)
    # print("Valid: %f" % (np.sum(bestids >=1) / bestids.shape[0]))
    # bestids = bestids[bestids >= 0]
    # unique_id = list()
    # counts = list()
    # for k in np.unique(bestids):
    #     unique_id.append(k)
    #     counts.append(np.sum(bestids == k))
    # plt.figure()
    # plt.plot(np.array(unique_id), np.array(counts) / bestids.shape[0])
    # plt.show()

def read_splits(iters):
    split_root = os.path.join(project_rootdir, 'exp_nyu_v2/splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'nyudepthv2_organized_train_files.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'nyudepthv2_organized_test_files.txt'), 'r')]
    return evaluation_entries

def train(processid, args, entries, iters=0):
    interval = np.floor(len(entries) / args.nprocs).astype(np.int).item()
    if processid == args.nprocs - 1:
        stidx = int(interval * processid)
        edidx = len(entries)
    else:
        stidx = int(interval * processid)
        edidx = int(interval * (processid + 1))

    eval_dataset = NYUV2(root=args.dataset_root, entries=entries[stidx : edidx],  flowPred_root=args.flowPred_root, mdPred_root=args.mdPred_root, iter=iters)
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=False, num_workers=args.num_workers, drop_last=False, shuffle=False)
    validate_RANSAC_odom_relpose(args, eval_loader, samplenum=args.samplenum, iters=iters)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--mdPred_root', type=str)
    parser.add_argument('--flowPred_root', type=str)
    parser.add_argument('--ins_root', type=str)
    parser.add_argument('--export_root', type=str)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--samplenum', type=int, default=50000)
    parser.add_argument('--nprocs', type=int, default=6)
    parser.add_argument('--dovls', action="store_true")
    parser.add_argument('--ban_scale_check', action="store_true")
    parser.add_argument('--stid', type=int, default=0)
    parser.add_argument('--edid', type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    for k in range(args.stid, args.edid):
        print("Start Iteration %d" % (k))
        entries = read_splits(k)
        mp.spawn(train, nprocs=args.nprocs, args=(args, entries, k))