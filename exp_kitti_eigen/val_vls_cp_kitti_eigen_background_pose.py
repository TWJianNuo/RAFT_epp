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
def validate_kitti(model, args, eval_loader, eppCbck, eppconcluer, group, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    epe_list = torch.zeros(2).cuda(device=args.gpu)
    out_list = torch.zeros(2).cuda(device=args.gpu)
    eppc_list = torch.zeros(2).cuda(device=args.gpu)

    mvl_list = torch.zeros(2).cuda(device=args.gpu)
    angl_list = torch.zeros(2).cuda(device=args.gpu)
    residual_opt_list = torch.zeros(2).cuda(device=args.gpu)
    residual_gt_list = torch.zeros(2).cuda(device=args.gpu)

    mvlv2d_list = torch.zeros(2).cuda(device=args.gpu)
    anglv2d_list = torch.zeros(2).cuda(device=args.gpu)
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

        pts1, pts2 = eppconcluer.flowmap2ptspair(flowmap=flow.unsqueeze(0), valid=semantic_selector.unsqueeze(0))
        outputsrec = eppconcluer.newton_gauss_F(pts2d1=pts1, pts2d2=pts2, intrinsic=intrinsic.squeeze(), posegt=rel_pose.squeeze(), posedeepv2d=rel_pose_deepv2d)

        # if args.gpu == 0:
        #     print("\nIteration: %d, loss_mv: %.2E, loss_ang: %.2E" % (val_id, outputsrec['loss_mv'], outputsrec['loss_ang']))

        if outputsrec['loss_mv'] > 0.1:
            continue

        if False:
            rel_pose_deepv2d[0:3, 3] = rel_pose_deepv2d[0:3, 3] / torch.norm(rel_pose_deepv2d[0:3, 3]) * torch.norm(rel_pose[0, 0:3, 3])
            poses = [rel_pose.squeeze().cpu().numpy(), rel_pose_deepv2d.squeeze().cpu().numpy(), outputsrec['pose_est'].squeeze().cpu().numpy()]
            vlsroots = ['/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/kitti_imu_eigen', '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/deepv2d_posevls_eigen',
                        '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/raft_posevls_eigen']

            seq, imgname, _ = batch['entry'][0].split(' ')

            image1_unpad = padder.unpad(image1[0])
            image2_unpad = padder.unpad(image2[0])
            img1 = image1_unpad.cpu().detach().permute([1, 2, 0]).numpy().astype(np.uint8)
            img2 = image2_unpad.cpu().detach().permute([1, 2, 0]).numpy().astype(np.uint8)

            depthnp = depth[0].cpu().squeeze().numpy()
            validnp = depthnp > 0

            h, w = image1_unpad.shape[1::]
            xx, yy = np.meshgrid(range(w), range(h), indexing='xy')

            for k in range(3):
                vlsroot = vlsroots[k]
                posec = poses[k]

                xxf = xx[validnp]
                yyf = yy[validnp]
                depthf = depthnp[validnp]

                pts3d = np.stack([xxf * depthf, yyf * depthf, depthf, np.ones_like(xxf)], axis=0)

                intrinsicnp = np.eye(4)
                intrinsicnp[0:3, 0:3] = intrinsic.squeeze().cpu().numpy()
                pts3d_oview = intrinsicnp @ posec @ np.linalg.inv(intrinsicnp) @ pts3d
                pts3d_oview_x = pts3d_oview[0, :] / pts3d_oview[2, :]
                pts3d_oview_y = pts3d_oview[1, :] / pts3d_oview[2, :]

                cm = plt.get_cmap('magma')
                vmax = 0.15
                tnp = 1 / depthf / vmax
                tnp = cm(tnp)

                fig = plt.figure(figsize=(16, 9))
                fig.add_subplot(2, 1, 1)
                plt.scatter(xxf, yyf, 1, tnp)
                plt.imshow(img1)

                fig.add_subplot(2, 1, 2)
                plt.scatter(pts3d_oview_x, pts3d_oview_y, 1, tnp)
                plt.imshow(img2)

                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                plt.savefig(os.path.join(vlsroot, "{}_{}.png".format(seq.split("/")[1], imgname.zfill(10))))
                plt.close()

        if False:
            exportroot = '/media/shengjie/disk1/Prediction/RAFT_eigen_pose'
            seq, imgname, _ = batch['entry'][0].split(' ')
            svfold = os.path.join(exportroot, seq)
            os.makedirs(svfold, exist_ok=True)

            picklepath = os.path.join(svfold, "{}.pickle".format(imgname.zfill(10)))
            pose2write = outputsrec['pose_est'].squeeze().cpu().numpy()
            with open(picklepath, 'wb') as handle:
                pickle.dump(pose2write, handle, protocol=pickle.HIGHEST_PROTOCOL)

            exportroot = '/media/shengjie/disk1/Prediction/IMU_eigen_pose'
            seq, imgname, _ = batch['entry'][0].split(' ')
            svfold = os.path.join(exportroot, seq)
            os.makedirs(svfold, exist_ok=True)

            picklepath = os.path.join(svfold, "{}.pickle".format(imgname.zfill(10)))
            pose2write = rel_pose.squeeze().cpu().numpy()
            with open(picklepath, 'wb') as handle:
                pickle.dump(pose2write, handle, protocol=pickle.HIGHEST_PROTOCOL)


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

        mvl_list[0] += outputsrec['loss_mv']
        mvl_list[1] += 1

        angl_list[0] += outputsrec['loss_ang']
        angl_list[1] += 1

        mvlv2d_list[0] += outputsrec['loss_mv_dv2d']
        mvlv2d_list[1] += 1

        anglv2d_list[0] += outputsrec['loss_ang_dv2d']
        anglv2d_list[1] += 1

        residual_opt_list[0] += outputsrec['loss_constrain']
        residual_opt_list[1] += 1

        residual_gt_list[0] += outputsrec['loss_constrain_gt']
        residual_gt_list[1] += 1

    if args.distributed:
        dist.all_reduce(tensor=epe_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=out_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=eppc_list, op=dist.ReduceOp.SUM, group=group)

        dist.all_reduce(tensor=mvl_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=angl_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=residual_opt_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=residual_gt_list, op=dist.ReduceOp.SUM, group=group)

        dist.all_reduce(tensor=mvlv2d_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=anglv2d_list, op=dist.ReduceOp.SUM, group=group)

    if args.gpu == 0:
        epe = epe_list[0] / epe_list[1]
        f1 = 100 * out_list[0] / out_list[1]
        eppc = eppc_list[0] / eppc_list[1]

        mvl = mvl_list[0] / mvl_list[1]
        angl = angl_list[0] / angl_list[1]
        residual_optl = residual_opt_list[0] / residual_opt_list[1]
        residual_gtl = residual_gt_list[0] / residual_gt_list[1]

        mvl_dv2d = mvlv2d_list[0] / mvlv2d_list[1]
        angl_dv2d = anglv2d_list[0] / anglv2d_list[1]

        return {'kitti-epe': float(epe.detach().cpu().numpy()), 'kitti-f1': float(f1.detach().cpu().numpy()), 'kitti-eppc': float(eppc.detach().cpu().numpy()),
                'mvl': float(mvl.detach().cpu().numpy()), 'angl': float(angl.detach().cpu().numpy()), 'mvl_dv2d': float(mvl_dv2d.detach().cpu().numpy()), 'angl_dv2d': float(angl_dv2d.detach().cpu().numpy()),
                'residual_optl': float(residual_optl.detach().cpu().numpy()), 'residual_gtl': float(residual_gtl.detach().cpu().numpy())
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

class eppConcluer(torch.nn.Module):
    def __init__(self, itnum=100, laplacian=1e-2, lr=0.1):
        super(eppConcluer, self).__init__()
        self.itnum = itnum
        self.laplacian = laplacian
        self.lr = lr

    def rot2ang(self, R):
        sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        ang = torch.zeros([3], device=R.device, dtype=torch.float)
        ang[0] = torch.atan2(R[2, 1], R[2, 2])
        ang[1] = torch.atan2(-R[2, 0], sy)
        ang[2] = torch.atan2(R[1, 0], R[0, 0])
        return ang

    def rot_from_axisangle(self, angs):
        """Convert an axisangle rotation into a 4x4 transformation matrix
        (adapted from https://github.com/Wallacoloo/printipi)
        Input 'vec' has to be Bx1x3
        """
        rotx = torch.eye(3, device=angs.device, dtype=torch.float)
        roty = torch.eye(3, device=angs.device, dtype=torch.float)
        rotz = torch.eye(3, device=angs.device, dtype=torch.float)

        rotx[1, 1] = torch.cos(angs[0])
        rotx[1, 2] = -torch.sin(angs[0])
        rotx[2, 1] = torch.sin(angs[0])
        rotx[2, 2] = torch.cos(angs[0])

        roty[0, 0] = torch.cos(angs[1])
        roty[0, 2] = torch.sin(angs[1])
        roty[2, 0] = -torch.sin(angs[1])
        roty[2, 2] = torch.cos(angs[1])

        rotz[0, 0] = torch.cos(angs[2])
        rotz[0, 1] = -torch.sin(angs[2])
        rotz[1, 0] = torch.sin(angs[2])
        rotz[1, 1] = torch.cos(angs[2])

        rot = rotz @ (roty @ rotx)
        return rot

    def t2T(self, t):
        T = torch.zeros([3, 3], device=t.device, dtype=torch.float)
        T[0, 1] = -t[2]
        T[0, 2] = t[1]
        T[1, 0] = t[2]
        T[1, 2] = -t[0]
        T[2, 0] = -t[1]
        T[2, 1] = t[0]
        return T

    def derivative_angle(self, angs):
        rotx = torch.eye(3, device=angs.device, dtype=torch.float)
        roty = torch.eye(3, device=angs.device, dtype=torch.float)
        rotz = torch.eye(3, device=angs.device, dtype=torch.float)

        rotx[1, 1] = torch.cos(angs[0])
        rotx[1, 2] = -torch.sin(angs[0])
        rotx[2, 1] = torch.sin(angs[0])
        rotx[2, 2] = torch.cos(angs[0])

        roty[0, 0] = torch.cos(angs[1])
        roty[0, 2] = torch.sin(angs[1])
        roty[2, 0] = -torch.sin(angs[1])
        roty[2, 2] = torch.cos(angs[1])

        rotz[0, 0] = torch.cos(angs[2])
        rotz[0, 1] = -torch.sin(angs[2])
        rotz[1, 0] = torch.sin(angs[2])
        rotz[1, 1] = torch.cos(angs[2])

        rotxd = torch.zeros([3, 3], device=angs.device, dtype=torch.float)
        rotyd = torch.zeros([3, 3], device=angs.device, dtype=torch.float)
        rotzd = torch.zeros([3, 3], device=angs.device, dtype=torch.float)

        rotxd[1, 1] = -torch.sin(angs[0])
        rotxd[1, 2] = -torch.cos(angs[0])
        rotxd[2, 1] = torch.cos(angs[0])
        rotxd[2, 2] = -torch.sin(angs[0])

        rotyd[0, 0] = -torch.sin(angs[1])
        rotyd[0, 2] = torch.cos(angs[1])
        rotyd[2, 0] = -torch.cos(angs[1])
        rotyd[2, 2] = -torch.sin(angs[1])

        rotzd[0, 0] = -torch.sin(angs[2])
        rotzd[0, 1] = -torch.cos(angs[2])
        rotzd[1, 0] = torch.cos(angs[2])
        rotzd[1, 1] = -torch.sin(angs[2])

        rotxd = rotz @ roty @ rotxd
        rotyd = rotz @ rotyd @ rotx
        rotzd = rotzd @ roty @ rotx

        return rotxd, rotyd, rotzd

    def derivative_translate(self, device):
        T0 = torch.zeros([3, 3], device=device, dtype=torch.float)
        T1 = torch.zeros([3, 3], device=device, dtype=torch.float)
        T2 = torch.zeros([3, 3], device=device, dtype=torch.float)

        T0[1, 2] = -1
        T0[2, 1] = 1

        T1[0, 2] = 1
        T1[2, 0] = -1

        T2[0, 1] = -1
        T2[1, 0] = 1

        return T0, T1, T2

    def compute_JacobianM(self, pts2d1, pts2d2, intrinsic, t, ang, lagr):
        R = self.rot_from_axisangle(ang)
        T = self.t2T(t)

        derT0, derT1, derT2 = self.derivative_translate(intrinsic.device)
        rotxd, rotyd, rotzd = self.derivative_angle(ang)

        pts2d1_bz = (pts2d1.T).unsqueeze(2)
        pts2d2_bz = (pts2d2.T).unsqueeze(2)
        samplenum = pts2d1.shape[1]

        r_bias = (torch.norm(t) - 1)
        J_t0_bias = 2 * lagr * r_bias / torch.norm(t) * t[0]
        J_t1_bias = 2 * lagr * r_bias / torch.norm(t) * t[1]
        J_t2_bias = 2 * lagr * r_bias / torch.norm(t) * t[2]

        pts2d2_bz_t = torch.transpose(torch.transpose(pts2d2_bz, 1, 2) @ torch.inverse(intrinsic).T, 1, 2)
        pts2d1_bz_t = torch.inverse(intrinsic) @ pts2d1_bz
        r = (torch.transpose(pts2d2_bz_t, 1, 2) @ T @ R @ pts2d1_bz_t).squeeze()
        derivM = pts2d2_bz_t @ torch.transpose(pts2d1_bz_t, 1, 2)

        J_t0 = torch.sum(derivM * (derT0 @ R), dim=[1, 2]) * 2 * r / samplenum + J_t0_bias / samplenum
        J_t1 = torch.sum(derivM * (derT1 @ R), dim=[1, 2]) * 2 * r / samplenum + J_t1_bias / samplenum
        J_t2 = torch.sum(derivM * (derT2 @ R), dim=[1, 2]) * 2 * r / samplenum + J_t2_bias / samplenum

        J_ang0 = torch.sum(derivM * (T @ rotxd), dim=[1, 2]) * 2 * r / samplenum
        J_ang1 = torch.sum(derivM * (T @ rotyd), dim=[1, 2]) * 2 * r / samplenum
        J_ang2 = torch.sum(derivM * (T @ rotzd), dim=[1, 2]) * 2 * r / samplenum

        JacobM = torch.stack([J_ang0, J_ang1, J_ang2, J_t0, J_t1, J_t2], dim=1)
        residual = (r ** 2 / samplenum + lagr * r_bias ** 2 / samplenum).unsqueeze(1)
        return JacobM, residual, r.abs().mean().detach()

    def flowmap2ptspair(self, flowmap, valid):
        _, _, h, w = flowmap.shape

        assert (flowmap.shape[0] == 1) and (valid.shape[0] == 1)

        flowmaps = flowmap.squeeze()
        valids = valid.squeeze() == 1
        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        xx = torch.from_numpy(xx).float().cuda(flowmap.device)
        yy = torch.from_numpy(yy).float().cuda(flowmap.device)

        xxf = xx[valids]
        yyf = yy[valids]

        flowxf = flowmaps[0, valids]
        flowyf = flowmaps[1, valids]

        xxf_o = xxf + flowxf
        yyf_o = yyf + flowyf

        pts1 = torch.stack([xxf, yyf, torch.ones_like(xxf)], dim=0)
        pts2 = torch.stack([xxf_o, yyf_o, torch.ones_like(xxf)], dim=0)

        return pts1, pts2

    def newton_gauss_F(self, pts2d1, pts2d2, intrinsic, posegt, posedeepv2d):
        # Newton Gauss Alg
        outputsrec = dict()

        ang_reg = torch.zeros([3], device=intrinsic.device, dtype=torch.float)
        t_reg = torch.zeros([3], device=intrinsic.device, dtype=torch.float)
        t_reg[-1] = -0.95

        minLoss = 1e10
        for kk in range(self.itnum):
            J, r, totloss = self.compute_JacobianM(pts2d1, pts2d2, intrinsic, t_reg, ang_reg, self.laplacian)
            curloss = r.sum().detach()

            try:
                M = J.T @ J
                inverseM = torch.inverse(M)

                if (inverseM @ M - torch.eye(6, device=intrinsic.device, dtype=torch.float)).abs().max() > 0.1:
                    break
                updatem = inverseM @ J.T @ r
            except:
                break

            if curloss > minLoss:
                break
            else:
                minLoss = curloss

            ang_reg = (ang_reg - self.lr * updatem[0:3, 0])
            t_reg = (t_reg - self.lr * updatem[3:6, 0])

        E_est = self.t2T(t_reg / torch.norm(t_reg)) @ self.rot_from_axisangle(ang_reg)
        F_est = torch.inverse(intrinsic).T @ E_est @ torch.inverse(intrinsic)
        loss_est = torch.mean(torch.abs(torch.sum((pts2d2.T @ F_est) * pts2d1.T, dim=1)))

        E_gt = self.t2T((posegt[0:3, 3]) / torch.norm(posegt[0:3, 3])) @ posegt[0:3, 0:3]
        F_gt = torch.inverse(intrinsic).T @ E_gt @ torch.inverse(intrinsic)
        loss_gt = torch.mean(torch.abs(torch.sum((pts2d2.T @ F_gt) * pts2d1.T, dim=1)))

        combinations = self.extract_RT_analytic(E_est)
        t_est, R_est = self.select_RT(combinations, pts2d1=pts2d1, pts2d2=pts2d2, intrinsic=intrinsic)

        t_est = (t_est / torch.norm(t_est))
        t_gt = (posegt[0:3, 3] / torch.norm(posegt[0:3, 3])).float()
        t_deepv2d = (posedeepv2d[0:3, 3] / torch.norm(posedeepv2d[0:3, 3])).float()

        loss_mv = 1 - torch.sum(t_est * t_gt)
        loss_ang = torch.mean((self.rot2ang((posegt[0:3, 0:3])) - self.rot2ang(R_est)).abs())

        loss_mv_dv2d = 1 - torch.sum(t_deepv2d * t_gt)
        loss_ang_dv2d = torch.mean((self.rot2ang((posegt[0:3, 0:3])) - self.rot2ang(posedeepv2d[0:3, 0:3])).abs())

        pose_est = torch.zeros_like(posegt)
        pose_est[0:3, 0:3] = self.rot_from_axisangle(ang_reg)
        pose_est[0:3, 3] = t_est * torch.norm(posegt[0:3, 3])

        outputsrec['loss_mv'] = float(loss_mv.detach().cpu().numpy())
        outputsrec['loss_ang'] = float(loss_ang.detach().cpu().numpy())
        outputsrec['loss_mv_dv2d'] = float(loss_mv_dv2d.detach().cpu().numpy())
        outputsrec['loss_ang_dv2d'] = float(loss_ang_dv2d.detach().cpu().numpy())
        outputsrec['loss_constrain'] = float(loss_est.detach().cpu().numpy())
        outputsrec['loss_constrain_gt'] = float(loss_gt.detach().cpu().numpy())
        outputsrec['pose_est'] = pose_est

        if kk <= 5:
            print("\nOptimization terminated early at iteration %d" % kk)
        return outputsrec

    def extract_RT_analytic(self, E):
        w = torch.zeros([3, 3], device=E.device, dtype=torch.float)
        w[0, 1] = -1
        w[1, 0] = 1
        w[2, 2] = 1

        M = E @ E.T
        t2 = torch.sqrt((M[0, 0] + M[1, 1] - M[2, 2]) / 2)
        t1 = torch.exp(torch.log(torch.abs((M[1, 2] + M[2, 1]) / 2)) - torch.log(torch.abs(t2))) * torch.sign(t2) * torch.sign(M[2, 1] + M[1, 2]) * (-1)
        t0 = torch.exp(torch.log(torch.abs((M[0, 2] + M[2, 0]) / 2)) - torch.log(torch.abs(t2))) * torch.sign(t2) * torch.sign(M[2, 0] + M[0, 2]) * (-1)
        recovert = torch.stack([t0, t1, t2])

        w1 = torch.cross(recovert, E[:, 0])
        w2 = torch.cross(recovert, E[:, 1])
        w3 = torch.cross(recovert, E[:, 2])

        r11 = w1 + torch.cross(w2, w3)
        r12 = w2 + torch.cross(w3, w1)
        r13 = w3 + torch.cross(w1, w2)
        recoverR1 = torch.stack([r11, r12, r13], dim=0).T

        r21 = w1 + torch.cross(w3, w2)
        r22 = w2 + torch.cross(w1, w3)
        r23 = w3 + torch.cross(w2, w1)
        recoverR2 = torch.stack([r21, r22, r23], dim=0).T

        if torch.det(recoverR1) < 0:
            recoverR1 = -recoverR1
        if torch.det(recoverR2) < 0:
            recoverR2 = -recoverR2

        combinations = [
            [recovert, recoverR1],
            [recovert, recoverR2],
            [-recovert, recoverR1],
            [-recovert, recoverR2]
        ]

        return combinations

    def select_RT(self, combinations, pts2d1, pts2d2, intrinsic):
        pospts = torch.zeros([4], device=intrinsic.device, dtype=torch.float)
        for idx, (tc, Rc) in enumerate(combinations):
            rdepth1 = self.flow2depth_relative(pts2d1, pts2d2, intrinsic, Rc, tc)
            rdepth2 = self.flow2depth_relative(pts2d2, pts2d1, intrinsic, Rc.T, -tc)
            pospts[idx] = torch.sum(rdepth1 > 0) + torch.sum(rdepth2 > 0)
        maxidx = torch.argmax(pospts)
        return combinations[maxidx]

    def flow2depth_relative(self, pts2d1, pts2d2, intrinsic, R, t):
        M = intrinsic @ R @ torch.inverse(intrinsic)
        delta_t = (intrinsic @ t.unsqueeze(1)).squeeze()

        denom = (pts2d2[0, :] * (M[2, :].unsqueeze(0) @ pts2d1).squeeze() - (M[0, :].unsqueeze(0) @ pts2d1).squeeze()) + \
                (pts2d2[1, :] * (M[2, :].unsqueeze(0) @ pts2d1).squeeze() - (M[1, :].unsqueeze(0) @ pts2d1).squeeze())
        rdepth = ((delta_t[0] - pts2d2[0, :] * delta_t[2]) + (delta_t[1] - pts2d2[1, :] * delta_t[2])) / denom
        return rdepth

    def flow2depth(self, pts2d1, pts2d2, intrinsic, R, t, coorespondedDepth):
        M = intrinsic @ R @ torch.inverse(intrinsic)
        delta_t = (intrinsic @ t.unsqueeze(1)).squeeze()
        minval = 1e-6

        denom = (pts2d2[0, :] * (M[2, :].unsqueeze(0) @ pts2d1).squeeze() - (M[0, :].unsqueeze(0) @ pts2d1).squeeze()) ** 2 + \
                (pts2d2[1, :] * (M[2, :].unsqueeze(0) @ pts2d1).squeeze() - (M[1, :].unsqueeze(0) @ pts2d1).squeeze()) ** 2

        selector = (denom > minval)
        denom = torch.clamp(denom, min=minval, max=np.inf)

        rel_d = torch.sqrt(
            ((delta_t[0] - pts2d2[0, :] * delta_t[2]) ** 2 +
             (delta_t[1] - pts2d2[1, :] * delta_t[2]) ** 2) / denom)
        alpha = torch.mean(coorespondedDepth) / torch.mean(rel_d)
        recover_d = alpha * rel_d

        return recover_d, alpha, selector

def train(gpu, ngpus_per_node, args):
    print("Using GPU %d for training" % gpu)
    args.gpu = gpu

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=ngpus_per_node, rank=args.gpu)

    model = RAFT(args)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(module=model)
        model = model.to(f'cuda:{args.gpu}')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True, output_device=args.gpu)

        eppCbck = eppConstrainer_background(height=args.image_size[0], width=args.image_size[1], bz=args.batch_size)
        eppCbck.to(f'cuda:{args.gpu}')

        eppconcluer = eppConcluer()
        eppconcluer.to(f'cuda:{args.gpu}')
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

    print(validate_kitti(model.module, args, eval_loader, eppCbck, eppconcluer, group))
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