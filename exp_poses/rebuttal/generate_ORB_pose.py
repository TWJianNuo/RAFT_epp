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
import matplotlib.pyplot as plt

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

class KITTI_eigen(data.Dataset):
    def __init__(self, entries, root='datasets/KITTI', odom_root=None, ins_root=None, flowPred_root=None, mdPred_root=None, bsposepred_root=None, banins=False):
        super(KITTI_eigen, self).__init__()
        self.root = root
        self.odom_root = odom_root
        self.flowPred_root = flowPred_root
        self.mdPred_root = mdPred_root
        self.ins_root = ins_root
        self.bsposepred_root = bsposepred_root
        self.banins = banins

        self.image_list = list()
        self.intrinsic_list = list()
        self.inspred_list = list()
        self.flowPred_list = list()
        self.mdPred_list = list()
        self.bsposepath_list = list()

        self.entries = list()

        for entry in self.remove_dup(entries):
            seq, index = entry.split(' ')
            index = int(index)

            if os.path.exists(os.path.join(root, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))):
                tmproot = root
            else:
                tmproot = odom_root

            img1path = os.path.join(tmproot, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))
            img2path = os.path.join(tmproot, seq, 'image_02', 'data', "{}.png".format(str(index + 1).zfill(10)))

            mdDepth_path = os.path.join(self.mdPred_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))
            if not os.path.exists(mdDepth_path):
                raise Exception("mD pred file %s missing" % mdDepth_path)

            flowpred_path = os.path.join(self.flowPred_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))

            # Load Intrinsic for each frame
            calib_dir = os.path.join(tmproot, seq.split('/')[0])

            cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
            velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
            imu2cam = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
            intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)

            if self.ins_root is not None:
                inspath = os.path.join(self.ins_root, seq, 'insmap/image_02', "{}.png".format(str(index).zfill(10)))
                if not os.path.exists(inspath):
                    raise Exception("instance file %s missing" % inspath)
                self.inspred_list.append(inspath)

            if not os.path.exists(img2path):
                self.image_list.append([img1path, img1path])
            else:
                self.image_list.append([img1path, img2path])

            pose_bs_path = os.path.join(self.bsposepred_root, seq, 'image_02/posepred', "{}.pickle".format(str(index).zfill(10)))
            if not os.path.exists(pose_bs_path):
                self.bsposepath_list.append('')
            else:
                self.bsposepath_list.append(pose_bs_path)

            self.intrinsic_list.append(intrinsic)
            self.entries.append(entry)
            self.flowPred_list.append(flowpred_path)
            self.mdPred_list.append(mdDepth_path)

        assert len(self.intrinsic_list) == len(self.entries) == len(self.image_list)

    def remove_dup(self, entries):
        dupentry = list()
        for entry in entries:
            seq, index, _ = entry.split(' ')
            dupentry.append("{} {}".format(seq, index.zfill(10)))

        removed = list(set(dupentry))
        removed.sort()
        return removed

    def __getitem__(self, index):
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        intrinsic = copy.deepcopy(self.intrinsic_list[index])
        if self.ins_root is not None:
            inspred = np.array(Image.open(self.inspred_list[index])).astype(np.int)
            if self.banins:
                inspred = inspred * 0
                inspred = inspred.astype(np.int)
        else:
            inspred = None

        if len(self.bsposepath_list[index]) == 0:
            posepred_bs = np.eye(4)
            posepred_bs = np.expand_dims(posepred_bs, axis=0)
            posepred_bs = np.repeat(posepred_bs, axis=0, repeats=int(inspred.max() + 1))
        else:
            posepred_bs = pickle.load(open(self.bsposepath_list[index], "rb"))

        mdDepth = np.array(Image.open(self.mdPred_list[index])).astype(np.float32) / 256.0
        flowpred_RAFT, valid_flow = readFlowKITTI(self.flowPred_list[index])
        data_blob = self.wrapup(img1=img1, img2=img2, intrinsic=intrinsic, insmap=inspred, mdDepth=mdDepth, flowpred_RAFT=flowpred_RAFT, posepred_bs=posepred_bs, tag=self.entries[index])
        return data_blob

    def wrapup(self, img1, img2, intrinsic, insmap, mdDepth, flowpred_RAFT, posepred_bs, tag):
        img1 = torch.from_numpy(img1).permute([2, 0, 1]).float()
        img2 = torch.from_numpy(img2).permute([2, 0, 1]).float()
        intrinsic = torch.from_numpy(intrinsic).float()
        mdDepth = torch.from_numpy(mdDepth).unsqueeze(0)
        flowpred_RAFT = torch.from_numpy(flowpred_RAFT).permute([2, 0, 1]).float()
        posepred_bs = torch.from_numpy(posepred_bs).float()

        data_blob = dict()
        data_blob['img1'] = img1
        data_blob['img2'] = img2
        data_blob['intrinsic'] = intrinsic
        data_blob['mdDepth'] = mdDepth
        data_blob['flowpred_RAFT'] = flowpred_RAFT
        data_blob['posepred_bs'] = posepred_bs
        data_blob['tag'] = tag

        if insmap is not None:
            insmap = torch.from_numpy(insmap).unsqueeze(0).int()
            data_blob['insmap'] = insmap

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

def inf_pose_flow(flow_pr_inf, insmap, optical_flow_orb_val, mdDepth, intrinsic, pid, gradComputer=None, banins=False, samplenum=50000, img1=None, img2=None):
    insmap_np = insmap[0, 0].cpu().numpy()
    intrinsicnp = intrinsic[0].cpu().numpy()
    optical_flow_orb_val_np = optical_flow_orb_val.squeeze().numpy()
    optical_flow_orb_val_np = optical_flow_orb_val_np == 1
    dummyh = 370
    gradbar = 0.9
    _, _, h, w = insmap.shape
    border_sel = np.zeros([h, w])
    border_sel[int(0.25810811 * dummyh) : int(0.99189189 * dummyh)] = 1
    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')

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
        selector = (xx_nf > 0) * (xx_nf < w) * (yy_nf > 0) * (yy_nf < h) * (insmap_np == 0) * border_sel * (depth_grad < gradbar) * optical_flow_orb_val_np
    else:
        selector = (xx_nf > 0) * (xx_nf < w) * (yy_nf > 0) * (yy_nf < h) * border_sel * (depth_grad < gradbar) * optical_flow_orb_val_np
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

    E, inliers = cv2.findEssentialMat(pts1, pts2, focal=intrinsicnp[0,0], pp=(intrinsicnp[0, 2], intrinsicnp[1, 2]), method=cv2.RANSAC, prob=0.9, threshold=0.5)
    cheirality_cnt, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=intrinsicnp[0, 0], pp=(intrinsicnp[0, 2], intrinsicnp[1, 2]))

    inliers_mask = inliers == 1
    inliers_mask = np.squeeze(inliers_mask, axis=1)
    pts1_inliers = pts1[inliers_mask, :].T
    pts2_inliers = pts2[inliers_mask, :].T

    pts1_inliers = np.concatenate([pts1_inliers, np.ones([1, pts1_inliers.shape[1]])], axis=0)
    pts2_inliers = np.concatenate([pts2_inliers, np.ones([1, pts2_inliers.shape[1]])], axis=0)
    scale_md = depth2scale(pts1_inliers, pts2_inliers, intrinsicnp, R, t, mdDepth_np[selector][rndidx][inliers_mask])

    if R[0, 0] < 0 or R[1, 1] < 0 or R[2, 2] < 0 or t[2] > 0 or (np.abs(t[2]) < np.abs(t[0])) or (np.abs(t[2]) < np.abs(t[1])):
        R = np.eye(3)
        t = np.array([[0, 0, -1]]).T
        scale_md = 0
        # print("Fail")
    # else:
    #     print("Success")

    # Image.fromarray(flow_to_image(flow_pr_inf[0].cpu().permute([1, 2, 0]).numpy())).show()
    # tensor2disp(torch.from_numpy(selector).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()
    # tensor2disp(depthmap > 0, vmax=1, viewind=0).show()
    # tensor2disp(torch.from_numpy(selvls).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()
    # fig = plt.figure(figsize=(16, 2.5))
    # fig.add_subplot(2, 1, 1)
    # plt.scatter(pts1_inliers[0, :], pts1_inliers[1, :], 3)
    # plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    #
    # fig.add_subplot(2, 1, 2)
    # plt.scatter(pts2_inliers[0, :], pts2_inliers[1, :], 3)
    # plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    #
    # plt.show()

    if np.sum(np.isnan(R)) + np.sum(np.isnan(t)) + np.sum(np.isnan(scale_md)) > 0:
        R = np.eye(3)
        t = np.array([[0, 0, -1]]).T
        scale_md = 0
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
def validate_RANSAC_odom_relpose(args, eval_loader, banins=False, bangrad=False, samplenum=50000, iters=0):
    if bangrad:
        gradComputer = None
    else:
        gradComputer = GradComputer()

    orb = cv2.ORB_create()

    for val_id, data_blob in enumerate(tqdm(eval_loader)):
    # for val_id, data_blob in enumerate((eval_loader)):
        insmap = data_blob['insmap']
        intrinsic = data_blob['intrinsic']
        flowpred = data_blob['flowpred_RAFT']
        mdDepth_pred = data_blob['mdDepth']
        pose_bs = data_blob['posepred_bs']
        tag = data_blob['tag'][0]

        _, _, h, w = mdDepth_pred.shape

        seq, frmidx = tag.split(' ')
        exportfold = os.path.join(args.export_root, str(iters).zfill(3), seq, 'image_02')
        os.makedirs(exportfold, exist_ok=True)
        export_root = os.path.join(exportfold, frmidx.zfill(10) + '.pickle')

        if torch.sum(torch.abs(data_blob['img1'] - data_blob['img2'])) < 1:
            R = np.eye(3)
            t = np.array([[0, 0, -1]]).T
            scale = 0
        else:
            img1 = cv2.cvtColor(copy.deepcopy(data_blob['img1'].squeeze().permute([1, 2, 0]).numpy().astype(np.uint8))[:, :, ::-1], cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(copy.deepcopy(data_blob['img2'].squeeze().permute([1, 2, 0]).numpy().astype(np.uint8))[:, :, ::-1], cv2.COLOR_BGR2GRAY)

            harr1 = cv2.cornerHarris(img1, 2, 3, 0.04)
            kp1_np = np.argwhere(harr1 > 0.01 * harr1.max())
            kp1 = [cv2.KeyPoint(float(x[1]), float(x[0]), 1) for x in kp1_np]

            harr2 = cv2.cornerHarris(img2, 2, 3, 0.04)
            kp2_np = np.argwhere(harr2 > 0.01 * harr2.max())
            kp2 = [cv2.KeyPoint(float(x[1]), float(x[0]), 1) for x in kp2_np]

            kp1, des1 = orb.compute(img1, kp1, None)
            kp2, des2 = orb.compute(img2, kp2, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(des1, des2)
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)

            optical_flow_orb = np.zeros([2, h, w])
            optical_flow_orb_val = np.zeros([h, w])
            for m in matches:
                xsrc, ysrc = kp1[m.queryIdx].pt
                xdst, ydst = kp2[m.trainIdx].pt
                optical_flow_orb[0, int(np.round(ysrc)), int(np.round(xsrc))] = xdst - xsrc
                optical_flow_orb[1, int(np.round(ysrc)), int(np.round(xsrc))] = ydst - ysrc
                optical_flow_orb_val[int(np.round(ysrc)), int(np.round(xsrc))] = 1
            optical_flow_orb_val = optical_flow_orb_val == 1
            optical_flow_orb = torch.from_numpy(optical_flow_orb).unsqueeze(0).float()
            optical_flow_orb_val = torch.from_numpy(optical_flow_orb_val)

            # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # plt.imshow(img3), plt.show()

            # plt.figure(), plt.imshow(data_blob['img1'].squeeze().permute([1, 2, 0]).cpu().numpy().astype(np.uint8))
            # ptsdraw = list()
            # for kkk in kp1:
            #     ptsdraw.append(np.array((kkk.pt[0], kkk.pt[1])))
            # ptsdraw = np.stack(ptsdraw, axis=0)
            # plt.scatter(ptsdraw[:, 0], ptsdraw[:, 1], 1, 'r')
            # plt.show()
            # flowpred
            # optical_flow_orb
            # print(optical_flow_orb[0, 0, optical_flow_orb_val == 1])
            # print(flowpred[0, 0, optical_flow_orb_val == 1])
            R, t, scale, _ = inf_pose_flow(optical_flow_orb, insmap, optical_flow_orb_val, mdDepth_pred, intrinsic, int(iters * eval_loader.__len__() + val_id), gradComputer=gradComputer, banins=banins, samplenum=samplenum, img1=img1, img2=img2)
        self_pose = np.eye(4)
        self_pose[0:3, 0:3] = R
        self_pose[0:3, 3:4] = t * scale

        pose_bs_np = pose_bs[0].cpu().numpy()
        pose_bs_np = pose_bs_np @ np.linalg.inv(pose_bs_np[0]) @ self_pose
        pose_bs_np[0] = self_pose

        with open(export_root, 'wb') as handle:
            pickle.dump(pose_bs_np, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_odomentries():
    seqmapping = \
    ['00 2011_10_03_drive_0027 000000 004540',
     "05 2011_09_30_drive_0018 000000 002760"
     ]

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

def train(processid, args, entries, iters=0):
    interval = np.floor(len(entries) / args.nprocs).astype(np.int).item()
    if processid == args.nprocs - 1:
        stidx = int(interval * processid)
        edidx = len(entries)
    else:
        stidx = int(interval * processid)
        edidx = int(interval * (processid + 1))

    eval_dataset = KITTI_eigen(root=args.dataset_root, odom_root=args.odom_root, entries=entries[stidx : edidx],  flowPred_root=args.flowPred_root,
                               mdPred_root=args.mdPred_root, ins_root=args.ins_root, bsposepred_root=args.bsposepred_root, banins=args.banins)
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=False, num_workers=args.num_workers, drop_last=False, shuffle=False)
    validate_RANSAC_odom_relpose(args, eval_loader, banins=False, bangrad=args.bangrad, samplenum=args.samplenum, iters=iters)
    return

def eval_generated_odom(args, seqmap, entries, repeats=64):
    accumerr = dict()
    pos_recs = dict()

    for val_id, entry in enumerate(tqdm(entries)):
        # Read Pose gt
        seq, frameidx, _ = entry.split(' ')
        seq = seq.split('/')[1]
        if seq not in accumerr.keys():
            accumerr[seq] = {'pose_RANSAC': list()}
            accumpos = {'pose_RANSAC': np.array([[0, 0, 0, 1]]).T, 'pose_gt': np.array([[0, 0, 0, 1]]).T}
            pos_recs[seq] = {'pose_RANSAC': list(), 'pose_gt': list()}

        frameidx = int(frameidx)
        gtposes_sourse = readlines(os.path.join(project_rootdir, 'exp_poses/kittiodom_gt/poses', "{}.txt".format(str(seqmap[seq[0:21]]['mapid']).zfill(2))))
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

        poses = dict()
        pose_RANSAC_path = os.path.join(args.export_root, str(repeats).zfill(3), entry.split(' ')[0], 'image_02', str(frameidx).zfill(10) + '.pickle')
        pose_RANSAC = pickle.load(open(pose_RANSAC_path, "rb"))
        poses['pose_gt'] = posegt
        poses['pose_RANSAC'] = pose_RANSAC[0]

        # Update Pose
        for k in accumpos.keys():
            accumpos[k] = poses[k] @ accumpos[k]
            pos_recs[seq][k].append(accumpos[k])

        for k in accumerr[seq].keys():
            loss_pos = np.sqrt(np.sum((accumpos[k] - accumpos['pose_gt']) ** 2))
            accumerr[seq][k].append(loss_pos.item())

    accumerr_fin = dict()
    for k in accumerr.keys():
        accumerr_fin[k] = dict()
        for kk in accumerr[k].keys():
            accumerr_fin[k][kk] = np.array(accumerr[k][kk])[-1] / len(accumerr[k][kk])

    weighted_err_dict = dict()
    for k in accumerr[seq].keys():
        totframe = 0
        for s in accumerr_fin.keys():
            totframe += len(accumerr[s][k])

        weighted_err = 0
        for s in accumerr_fin.keys():
            weighted_err += len(accumerr[s][k]) / totframe * accumerr_fin[s][k]

        weighted_err_dict[k] = weighted_err

    for k in weighted_err_dict.keys():
        print("%s : %f" % (k.ljust(50), weighted_err_dict[k]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--odom_root', type=str)
    parser.add_argument('--mdPred_root', type=str)
    parser.add_argument('--bsposepred_root', type=str)
    parser.add_argument('--flowPred_root', type=str)
    parser.add_argument('--ins_root', type=str)
    parser.add_argument('--export_root', type=str)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--banins', action='store_true')
    parser.add_argument('--bangrad', action='store_true')
    parser.add_argument('--only_eval', action='store_true')
    parser.add_argument('--ban_odometry', action='store_true')
    parser.add_argument('--samplenum', type=int, default=50000)
    parser.add_argument('--nprocs', type=int, default=6)
    parser.add_argument('--export_first_it', action='store_true')
    parser.add_argument('--delay', type=int, default=0)
    parser.add_argument('--stid', type=int, default=1)
    parser.add_argument('--edid', type=int, default=1)
    parser.add_argument('--evalonly', action='store_true')
    parser.add_argument('--skipexist', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    time.sleep(args.delay)

    k = 0
    seqmap, entries = get_odomentries()
    mp.spawn(train, nprocs=args.nprocs, args=(args, entries, k))