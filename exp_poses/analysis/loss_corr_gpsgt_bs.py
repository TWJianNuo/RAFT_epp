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
from core.utils.frame_utils import readFlowKITTI
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import glob
from exp_kitti_eigen_fixation.dataset_kitti_eigen_fixation import get_pose

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
    def __init__(self, entries, entries_map, root='datasets/KITTI', odom_root=None, ins_root=None, flowPred_root=None,
                 mdPred_root=None, bsposepred_root=None, flowgt_root=None, stereogt_root=None, RANSACPose_root=None, raw_root=None):
        super(KITTI_eigen, self).__init__()
        self.root = root
        self.odom_root = odom_root
        self.flowPred_root = flowPred_root
        self.mdPred_root = mdPred_root
        self.ins_root = ins_root
        self.bsposepred_root = bsposepred_root
        self.flowgt_root = flowgt_root
        self.stereogt_root = stereogt_root
        self.RANSACPose_root = RANSACPose_root
        self.raw_root = raw_root
        self.entries_map = entries_map

        self.image_list = list()
        self.intrinsic_list = list()
        self.inspred_list = list()
        self.flowPred_list = list()
        self.mdPred_list = list()
        self.flowGt_list = list()
        self.stereoGt_list = list()

        self.entries = list()

        self.RANSACPose_rec = list()
        self.IMUPose_rec = list()

        for idx, entry in enumerate(entries):
            seq, index, _ = entry.split(' ')
            index = int(index)

            if os.path.exists(os.path.join(root, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))):
                tmproot = root
            else:
                tmproot = odom_root

            img1path = os.path.join(tmproot, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))
            img2path = os.path.join(tmproot, seq, 'image_02', 'data', "{}.png".format(str(index + 1).zfill(10)))

            if not os.path.exists(img2path):
                self.image_list.append([img1path, img1path])
            else:
                self.image_list.append([img1path, img2path])

            mdDepth_path = os.path.join(self.mdPred_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))
            if not os.path.exists(mdDepth_path):
                raise Exception("mD pred file %s missing" % mdDepth_path)

            stereogt_path = os.path.join(self.stereogt_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))
            self.stereoGt_list.append(stereogt_path)

            flowpred_path = os.path.join(self.flowPred_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))
            flowgt_path = os.path.join(self.flowgt_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))

            # Load Intrinsic for each frame
            calib_dir = os.path.join(tmproot, seq.split('/')[0])

            cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
            velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
            imu2cam = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
            intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)

            # Read all Ranscac poses
            seq_raw, index_raw, _ = entries_map[idx].split(' ')
            poses_paths = glob.glob(os.path.join(self.RANSACPose_root, seq_raw, 'image_02', '*.pickle'))
            RANSACPose_seq = list()
            for k in range(len(poses_paths)):
                ps = os.path.join(self.RANSACPose_root, seq_raw, 'image_02', '{}.pickle'.format(str(k).zfill(10)))
                tmpPose = pickle.load(open(ps, "rb"))
                RANSACPose_seq.append(tmpPose)

            # Read all GPS Locations
            imupose_list = list()
            for k in range(len(poses_paths)):
                if k == len(poses_paths) - 1:
                    imupose_list.append(np.eye(4))
                else:
                    imupose_list.append(get_pose(raw_root, seq_raw, int(k), extrinsic))
            self.IMUPose_rec.append(imupose_list)
            self.RANSACPose_rec.append(RANSACPose_seq)

            if self.ins_root is not None:
                inspath = os.path.join(self.ins_root, seq, 'insmap/image_02', "{}.png".format(str(index).zfill(10)))
                if not os.path.exists(inspath):
                    raise Exception("instance file %s missing" % inspath)
                self.inspred_list.append(inspath)

            self.intrinsic_list.append(intrinsic)
            self.entries.append(entry)
            self.flowPred_list.append(flowpred_path)
            self.flowGt_list.append(flowgt_path)
            self.mdPred_list.append(mdDepth_path)

        assert len(self.intrinsic_list) == len(self.entries) == len(self.image_list)

    def __getitem__(self, index):
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        intrinsic = copy.deepcopy(self.intrinsic_list[index])
        RANSACPose = copy.deepcopy(self.RANSACPose_rec[index])
        IMUPose = copy.deepcopy(self.IMUPose_rec[index])
        if self.ins_root is not None:
            inspred = np.array(Image.open(self.inspred_list[index])).astype(np.int)
        else:
            inspred = None

        mdDepth = np.array(Image.open(self.mdPred_list[index])).astype(np.float32) / 256.0
        stereogt = np.array(Image.open(self.stereoGt_list[index])).astype(np.float32) / 256.0
        flowpred_RAFT, _ = readFlowKITTI(self.flowPred_list[index])
        flowgt, flowgt_valid = readFlowKITTI(self.flowGt_list[index])
        data_blob = self.wrapup(img1=img1, img2=img2, intrinsic=intrinsic, insmap=inspred, mdDepth=mdDepth, stereogt=stereogt,
                                flowpred_RAFT=flowpred_RAFT, flowgt=flowgt, flowgt_valid=flowgt_valid, RANSACPose=RANSACPose, IMUPose=IMUPose, tag=self.entries_map[index])
        return data_blob

    def wrapup(self, img1, img2, intrinsic, insmap, mdDepth, stereogt, flowpred_RAFT, flowgt, flowgt_valid, RANSACPose, IMUPose, tag):
        img1 = torch.from_numpy(img1).permute([2, 0, 1]).float()
        img2 = torch.from_numpy(img2).permute([2, 0, 1]).float()
        intrinsic = torch.from_numpy(intrinsic).float()
        mdDepth = torch.from_numpy(mdDepth).unsqueeze(0)
        stereogt = torch.from_numpy(stereogt).unsqueeze(0)
        flowpred_RAFT = torch.from_numpy(flowpred_RAFT).permute([2, 0, 1]).float()
        flowgt = torch.from_numpy(flowgt).permute([2, 0, 1]).float()
        flowgt_valid = torch.from_numpy(flowgt_valid).unsqueeze(0)

        data_blob = dict()
        data_blob['img1'] = img1
        data_blob['img2'] = img2
        data_blob['intrinsic'] = intrinsic
        data_blob['mdDepth'] = mdDepth
        data_blob['stereogt'] = stereogt
        data_blob['flowpred_RAFT'] = flowpred_RAFT
        data_blob['flowgt'] = flowgt
        data_blob['flowgt_valid'] = flowgt_valid
        data_blob['tag'] = tag
        data_blob['RANSACPose'] = RANSACPose
        data_blob['IMUPose'] = IMUPose

        if insmap is not None:
            insmap = torch.from_numpy(insmap).unsqueeze(0).int()
            data_blob['insmap'] = insmap

        return data_blob

    def __len__(self):
        return len(self.entries)

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

def inf_pose_flow(flow_pr_inf, insmap, mdDepth, intrinsic, pid, gradComputer=None, banins=False, samplenum=50000):
    insmap_np = insmap[0, 0].cpu().numpy()
    intrinsicnp = intrinsic[0].cpu().numpy()
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
        selector = (xx_nf > 0) * (xx_nf < w) * (yy_nf > 0) * (yy_nf < h) * (insmap_np == 0) * border_sel * (depth_grad < gradbar)
    else:
        selector = (xx_nf > 0) * (xx_nf < w) * (yy_nf > 0) * (yy_nf < h) * border_sel * (depth_grad < gradbar)
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

def get_coord_loss(IMUPose, RANSACPose, idx, estpose, span=50):
    pos = np.array([[0, 0, 0, 1]]).T
    imulocrec = list()
    for k in range(-span, span):
        curidx = idx + k
        if curidx >= len(IMUPose) - 2 or curidx < 0:
            continue
        pos = IMUPose[curidx].squeeze().numpy() @ pos
        imulocrec.append(pos)

    tmpRANSACPose = copy.deepcopy(RANSACPose)
    tmpRANSACPose[idx] = estpose.float().unsqueeze(0)

    pos = np.array([[0, 0, 0, 1]]).T
    RANSAClocrec = list()
    for k in range(-span, span):
        curidx = idx + k
        if curidx >= len(IMUPose) - 2 or curidx < 0:
            continue
        pos = tmpRANSACPose[curidx].squeeze().numpy() @ pos
        RANSAClocrec.append(pos)

    accumerr = 0
    for m in range(len(RANSAClocrec)):
        accumerr += np.sqrt(np.sum((RANSAClocrec[m][0:3, 0] - imulocrec[m][0:3, 0]) ** 2))

    RANSAClocrec_np = np.array(RANSAClocrec)[:, :, 0]
    imulocrec_np = np.array(imulocrec)[:, :, 0]
    plt.figure()
    plt.scatter(RANSAClocrec_np[:, 0], RANSAClocrec_np[:, 1], 5)
    plt.scatter(imulocrec_np[:, 0], imulocrec_np[:, 1], 5)
    plt.scatter(RANSAClocrec_np[idx, 0], RANSAClocrec_np[idx, 1], 5, 'r')
    plt.scatter(imulocrec_np[idx, 0], imulocrec_np[idx, 1], 5, 'r')
    plt.axis('equal')
    plt.legend(['RANSAC', 'IMU'])
    plt.savefig('/home/shengjie/Desktop/t1.png')
    plt.close()

    RANSAClocrec_np_aligned = copy.deepcopy(RANSAClocrec_np)
    RANSAClocrec_np_aligned[:, 0] = RANSAClocrec_np_aligned[:, 0] - RANSAClocrec_np_aligned[idx, 0]
    RANSAClocrec_np_aligned[:, 1] = RANSAClocrec_np_aligned[:, 1] - RANSAClocrec_np_aligned[idx, 1]
    imulocrec_np_aligned = copy.deepcopy(imulocrec_np)
    imulocrec_np_aligned[:, 0] = imulocrec_np_aligned[:, 0] - imulocrec_np_aligned[idx, 0]
    imulocrec_np_aligned[:, 1] = imulocrec_np_aligned[:, 1] - imulocrec_np_aligned[idx, 1]
    plt.figure()
    plt.scatter(RANSAClocrec_np_aligned[:, 0], RANSAClocrec_np_aligned[:, 1], 5)
    plt.scatter(imulocrec_np_aligned[:, 0], imulocrec_np_aligned[:, 1], 5)
    plt.scatter(RANSAClocrec_np_aligned[idx, 0], RANSAClocrec_np_aligned[idx, 1], 5, 'r')
    plt.scatter(imulocrec_np_aligned[idx, 0], imulocrec_np_aligned[idx, 1], 5, 'r')
    plt.axis('equal')
    plt.legend(['RANSAC', 'IMU'])
    plt.savefig('/home/shengjie/Desktop/t2.png')
    plt.close()

    return accumerr

def validate_RANSAC_odom_corr_bs(args, eval_loader, banins=False, bangrad=False, samplenum=50000):
    if bangrad:
        gradComputer = None
    else:
        gradComputer = GradComputer()

    trialtime = 32

    os.makedirs(args.vls_root, exist_ok=True)

    corrlist_ang = list()
    corrlist_t = list()
    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        insmap = data_blob['insmap']
        intrinsic = data_blob['intrinsic']
        flowpred = data_blob['flowpred_RAFT']
        mdDepth_pred = data_blob['mdDepth']
        tag = data_blob['tag'][0]
        flowgt = data_blob['flowgt']
        rgb1 = data_blob['img1'] / 255.0
        rgb2 = data_blob['img2'] / 255.0
        insselector = (insmap == 0).float()
        IMUPose = data_blob['IMUPose']
        RANSACPose = data_blob['RANSACPose']
        stereogt = data_blob['stereogt']
        flowgt_valid = data_blob['flowgt_valid'] * insselector * (stereogt > 0).float()

        self_poses = list()
        breakflag = False
        for k in range(trialtime):
            R, t, scale, _ = inf_pose_flow(flowpred, insmap, mdDepth_pred, intrinsic, int(val_id * 500 + k), gradComputer=gradComputer, banins=banins, samplenum=samplenum)
            if scale == 0:
                breakflag = True
            else:
                self_pose = np.eye(4)
                self_pose[0:3, 0:3] = R
                self_pose[0:3, 3:4] = t * scale
                self_pose = torch.from_numpy(self_pose).float()
            self_poses.append(self_pose)
        if breakflag:
            continue

        curidx = int(tag.split(' ')[1])

        ang_loss_rec = list()
        t_loss_rec = list()
        flow_loss_rec = list()
        for k in range(trialtime):
            self_pose = self_poses[k]
            self_ang = rot2ang(self_pose[0:3, 0:3])
            imu_ang = rot2ang(IMUPose[curidx][0, 0:3, 0:3])

            mdDepth_pred_sqz = stereogt.squeeze()
            _, _, h, w = insmap.shape
            xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
            xxt = torch.from_numpy(xx)
            yyt = torch.from_numpy(yy)
            ones = torch.ones_like(xxt)
            pts3d = torch.stack([xxt * mdDepth_pred_sqz, yyt * mdDepth_pred_sqz, mdDepth_pred_sqz, ones], dim=2).unsqueeze(-1)

            intrinsic_sqz = intrinsic.squeeze()
            projM = intrinsic_sqz @ self_pose @ torch.inverse(intrinsic_sqz)
            projM_ex = projM.unsqueeze(0).unsqueeze(0).expand([h, w, -1, -1])
            prj2d = projM_ex @ pts3d
            prj2d_x = prj2d[:, :, 0] / prj2d[:, :, 2]
            prj2d_y = prj2d[:, :, 1] / prj2d[:, :, 2]

            flowx = prj2d_x.squeeze() - xxt
            flowy = prj2d_y.squeeze() - yyt
            flowpred_depth = torch.stack([flowx, flowy], dim=0).unsqueeze(0)

            flow_loss = torch.sum(torch.abs(flowpred_depth - flowgt), dim=1, keepdim=True)
            flow_loss = torch.sum(flow_loss * flowgt_valid) / torch.sum(flowgt_valid)

            self_t = self_pose[0:3, 3:4]
            imu_t = IMUPose[curidx][0, 0:3, 3:4]

            ang_loss_rec.append(torch.abs(self_ang - imu_ang).mean().item())
            t_loss_rec.append(torch.abs(self_t - imu_t).mean().item())
            flow_loss_rec.append(flow_loss.item())

        # fig = plt.figure()
        # fig.add_subplot(2, 1, 1)
        # plt.scatter(flow_loss_rec, ang_loss_rec)
        # fig.add_subplot(2, 1, 2)
        # plt.scatter(flow_loss_rec, t_loss_rec)

        r_ang, _ = pearsonr(np.array(flow_loss_rec), np.array(ang_loss_rec))
        r_t, _ = pearsonr(np.array(flow_loss_rec), np.array(t_loss_rec))
        corrlist_ang.append(r_ang)
        corrlist_t.append(r_t)

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.hist(corrlist_ang, bins=100, range=(-1, 1))
    plt.title("Average Angular Coorelation: %f" % (np.mean(np.array(corrlist_ang))))
    plt.xlabel('Pearson Coorelation between l1 angular loss and flow loss')
    plt.ylabel('Num of Frames')
    fig.add_subplot(1, 2, 2)
    plt.hist(corrlist_t, bins=100, range=(-1, 1))
    plt.title("Average Scale Coorelation: %f" % (np.mean(np.array(corrlist_t))))
    plt.xlabel('Pearson Coorelation between l1 scale loss and flow loss')
    plt.ylabel('Num of Frames')
    plt.savefig(os.path.join(args.vls_root, 'coor_bs.png'))
    plt.close()
    print("Ang coor: %f, t coor: %f" % (np.mean(np.array(corrlist_ang)), np.mean(np.array(corrlist_t))))

def get_odomentries(args):
    import glob
    odomentries = list()
    odomseqs = [
        '2011_10_03/2011_10_03_drive_0027_sync',
        '2011_09_30/2011_09_30_drive_0016_sync',
        '2011_09_30/2011_09_30_drive_0018_sync',
        '2011_09_30/2011_09_30_drive_0027_sync'
    ]
    for odomseq in odomseqs:
        leftimgs = glob.glob(os.path.join(args.odom_root, odomseq, 'image_02/data', "*.png"))
        for leftimg in leftimgs:
            imgname = os.path.basename(leftimg)
            odomentries.append("{} {} {}".format(odomseq, imgname.rstrip('.png'), 'l'))
    return odomentries

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

def read_splits(args):
    split_root = os.path.join(project_rootdir, 'exp_pose_mdepth_kitti_eigen/splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'train_files.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'test_files.txt'), 'r')]
    odom_entries = get_odomentries(args)

    if args.only_eval:
        return evaluation_entries
    else:
        if args.ban_odometry:
            return train_entries + evaluation_entries
        else:
            return train_entries + evaluation_entries + odom_entries

def readlines(filename):
    with open(filename, 'r') as f:
        filenames = f.readlines()
    return filenames

def read_splits_mapping(args):
    evaluation_entries = list()
    expandentries = list()
    mappings = readlines(args.mpf_root)
    for idx, m in enumerate(mappings):
        if len(m) == 1:
            continue
        d, s, cidx = m.split(' ')
        seq = "{}/{}".format(d, s)
        expandentries.append("{} {} {}".format(seq, cidx.rstrip('\n'), 'l'))

        seqname = "kittistereo15_{}/kittistereo15_{}_sync".format(str(idx).zfill(6), str(idx).zfill(6))
        evaluation_entries.append("{} {} {}".format(seqname, "10".zfill(10), 'l'))
    return evaluation_entries, expandentries

def train(args, entries, entries_map):
    eval_dataset = KITTI_eigen(root=args.dataset_root, odom_root=args.odom_root, entries=entries, entries_map=entries_map, flowPred_root=args.flowPred_root, flowgt_root=args.flowgt_root,
                               mdPred_root=args.mdPred_root, ins_root=args.ins_root, bsposepred_root=args.bsposepred_root, stereogt_root=args.stereogt_root,
                               RANSACPose_root=args.RANSACPose_root, raw_root=args.raw_root)
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=True, num_workers=args.num_workers, drop_last=False, shuffle=False)
    validate_RANSAC_odom_corr_bs(args, eval_loader, banins=args.banins, bangrad=args.bangrad, samplenum=args.samplenum)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--odom_root', type=str)
    parser.add_argument('--mdPred_root', type=str)
    parser.add_argument('--bsposepred_root', type=str)
    parser.add_argument('--flowPred_root', type=str)
    parser.add_argument('--flowgt_root', type=str)
    parser.add_argument('--raw_root', type=str)
    parser.add_argument('--RANSACPose_root', type=str)
    parser.add_argument('--ins_root', type=str)
    parser.add_argument('--vls_root', type=str)
    parser.add_argument('--mpf_root', type=str)
    parser.add_argument('--stereogt_root', type=str)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--banins', action='store_true')
    parser.add_argument('--bangrad', action='store_true')
    parser.add_argument('--only_eval', action='store_true')
    parser.add_argument('--ban_odometry', action='store_true')
    parser.add_argument('--samplenum', type=int, default=50000)
    parser.add_argument('--nprocs', type=int, default=6)
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    entries, entries_map = read_splits_mapping(args)
    train(args, entries, entries_map)