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


def sequence_flowloss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def t2T(t):
    bz = t.shape[0]
    T = torch.zeros([bz, 3, 3], device=t.get_device())
    T[:, 0, 1] = -t[:, 2]
    T[:, 0, 2] = t[:, 1]
    T[:, 1, 0] = t[:, 2]
    T[:, 1, 2] = -t[:, 0]
    T[:, 2, 0] = -t[:, 1]
    T[:, 2, 1] = t[:, 0]
    return T

def pose2RT(rel_pose):
    R = rel_pose[:, 0:3, 0:3]
    T = t2T(rel_pose[:, 0:3, 3] / torch.norm(rel_pose[:, 0:3, 3], dim=1, keepdim=True).expand([-1, 3]))
    return T, R

def sequence_eppcloss(eppCbck, flow_preds, semantic_selector, intrinsic, rel_pose, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds)
    eppc_loss = 0.0

    metrics = dict()
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        eppc = eppCbck.epp_constrain(flowest=flow_preds[i], intrinsic=intrinsic, rel_pose=rel_pose, valid=semantic_selector)
        eppc_loss += i_weight * eppc

        if i == len(flow_preds) - 1:
            metrics['eppc'] = eppc.item()

    return eppc_loss, metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler, logpath, endstep):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.logpath = logpath
        self.cm = plt.get_cmap('magma')
        self.writer = None
        self.stt = time.time()
        self.endstep = endstep

    def create_summarywriter(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.logpath)

    def _print_training_status(self):
        avetime = (time.time() - self.stt) / self.total_steps * (self.endstep - self.total_steps) / 60 / 60
        training_str = "[step:{:6d}, lr:{:10.7f}, time:{:2.3f}h]: ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0], avetime)
        metrics_str = ''
        for k in sorted(self.running_loss.keys()):
            metrics_str += " {}: {:0.6f} |".format(k, self.running_loss[k] / SUM_FREQ)

        # print the training status
        print(training_str + metrics_str)

        self.create_summarywriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics, image1, image2, flowgt, flow_predictions, valid, depth):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.write_vls(image1, image2, flowgt, flow_predictions, valid, depth)
            self.running_loss = {}

    def write_vls(self, image1, image2, flowgt, flow_predictions, valid, depth):
        img1 = image1[0].detach().cpu().detach().permute([1, 2, 0]).numpy().astype(np.uint8)
        img2 = image2[0].detach().cpu().detach().permute([1, 2, 0]).numpy().astype(np.uint8)

        validnp = valid[0].detach().cpu().numpy() == 1
        depthnp = depth[0].squeeze().numpy()

        flow_pred = flow_to_image(flow_predictions[-1][0].permute([1, 2, 0]).detach().cpu().numpy(), rad_max=150)
        flow_gt = flow_to_image(flowgt[0].permute([1, 2, 0]).detach().cpu().numpy(), rad_max=150)

        h, w = image1.shape[2::]
        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        pixelloc = np.stack([xx, yy], axis=0)
        pixelloc = torch.from_numpy(pixelloc).float() + flow_predictions[-1][0].detach().cpu()
        pixelloc[0, :, :] = ((pixelloc[0, :, :] / w) - 0.5) * 2
        pixelloc[1, :, :] = ((pixelloc[1, :, :] / h) - 0.5) * 2
        pixelloc = pixelloc.permute([1, 2, 0])

        xxf = xx[validnp]
        yyf = yy[validnp]
        depthf = depthnp[validnp]
        xxf_oview = flowgt[0, 0].detach().cpu().numpy()[validnp] + xxf
        yyf_oview = flowgt[0, 1].detach().cpu().numpy()[validnp] + yyf

        vmax = 0.15
        tnp = 1 / depthf / vmax
        tnp = self.cm(tnp)

        fig = plt.figure(figsize=(16, 2.5))
        canvas = FigureCanvasAgg(fig)
        fig.add_subplot(1, 2, 1)
        plt.scatter(xxf, yyf, 1, tnp)
        plt.imshow(img1)

        fig.add_subplot(1, 2, 2)
        plt.scatter(xxf_oview, yyf_oview, 1, tnp)
        plt.imshow(img2)

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        canvas.draw()
        buf = canvas.buffer_rgba()
        plt.close()
        X = np.asarray(buf)
        X = np.array(Image.fromarray(X).resize([w * 2, h], Image.BILINEAR))

        image1_recon = torch.nn.functional.grid_sample(image2[0, :, :, :].detach().unsqueeze(0).cpu(), pixelloc.unsqueeze(0), mode='bilinear', align_corners=False)
        image1_recon = image1_recon[0].permute([1, 2, 0]).numpy().astype(np.uint8)

        img_mid = np.concatenate([flow_pred, flow_gt], axis=1)
        img_down = np.concatenate([img1, image1_recon], axis=1)
        img_vls = np.concatenate([X[:, :, 0:3], img_mid, img_down], axis=0)

        self.writer.add_image('img_vls', (torch.from_numpy(img_vls).float() / 255).permute([2, 0, 1]), self.total_steps)

    def write_dict(self, results, step):
        self.create_summarywriter()

        for key in results:
            self.writer.add_scalar(key, results[key], step)

    def close(self):
        self.writer.close()

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

        semantic_selector = batch['semantic_selector']
        semantic_selector = Variable(semantic_selector)
        semantic_selector = semantic_selector.cuda(args.gpu, non_blocking=True)

        depth = batch['depth']
        depth = Variable(depth)
        depth = depth.cuda(args.gpu, non_blocking=True)

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0])

        pts1, pts2 = eppconcluer.flowmap2ptspair(flowmap=flow.unsqueeze(0), valid=semantic_selector.unsqueeze(0))
        outputsrec = eppconcluer.newton_gauss_F(pts2d1=pts1, pts2d2=pts2, intrinsic=intrinsic.squeeze(), posegt=rel_pose.squeeze())

        # if outputsrec['loss_mv'] > 0.01 and outputsrec['loss_mv'] < 0.1:
        #     image1_unpad = padder.unpad(image1[0])
        #     image2_unpad = padder.unpad(image2[0])
        #     img1 = image1_unpad.cpu().detach().permute([1, 2, 0]).numpy().astype(np.uint8)
        #     img2 = image2_unpad.cpu().detach().permute([1, 2, 0]).numpy().astype(np.uint8)
        #
        #     validnp = valid_gt.detach().cpu().numpy() == 1
        #     depthnp = depth[0].cpu().squeeze().numpy()
        #
        #     h, w = image1_unpad.shape[1::]
        #     xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        #
        #     xxf = xx[validnp]
        #     yyf = yy[validnp]
        #     depthf = depthnp[validnp]
        #     xxf_oview = flow_gt[0].detach().cpu().numpy()[validnp] + xxf
        #     yyf_oview = flow_gt[1].detach().cpu().numpy()[validnp] + yyf
        #
        #     cm = plt.get_cmap('magma')
        #     vmax = 0.15
        #     tnp = 1 / depthf / vmax
        #     tnp = cm(tnp)
        #
        #     fig = plt.figure(figsize=(16, 2.5))
        #     canvas = FigureCanvasAgg(fig)
        #     fig.add_subplot(1, 2, 1)
        #     plt.scatter(xxf, yyf, 1, tnp)
        #     plt.imshow(img1)
        #
        #     fig.add_subplot(1, 2, 2)
        #     plt.scatter(xxf_oview, yyf_oview, 1, tnp)
        #     plt.imshow(img2)
        #
        #     plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        #     canvas.draw()
        #     buf = canvas.buffer_rgba()
        #     plt.close()
        #     X = np.asarray(buf)
        #     X = np.array(Image.fromarray(X).resize([w * 2, h], Image.BILINEAR))
        #     Image.fromarray(X).show()

        if outputsrec['loss_mv'] > 0.1:
            continue

        eppc = eppCbck.epp_constrain_val(flowest=flow, intrinsic=intrinsic, rel_pose=rel_pose, valid=semantic_selector)
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

    if args.gpu == 0:
        epe = epe_list[0] / epe_list[1]
        f1 = 100 * out_list[0] / out_list[1]
        eppc = eppc_list[0] / eppc_list[1]

        mvl = mvl_list[0] / mvl_list[1]
        angl = angl_list[0] / angl_list[1]
        residual_optl = residual_opt_list[0] / residual_opt_list[1]
        residual_gtl = residual_gt_list[0] / residual_gt_list[1]

        # print("Validation KITTI, epe: %f, f1: %f, eppc: %f, mvl: %f, angl: %f, residual_optl: %f, residual_gt: %f" % (epe, f1, eppc, mvl, angl, residual_optl, residual_gtl))
        return {'kitti-epe': float(epe.detach().cpu().numpy()), 'kitti-f1': float(f1.detach().cpu().numpy()), 'kitti-eppc': float(eppc.detach().cpu().numpy()),
                'mvl': float(mvl.detach().cpu().numpy()), 'angl': float(angl.detach().cpu().numpy()),
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
    def __init__(self, height, width):
        super(eppConstrainer_background, self).__init__()

        self.height = height
        self.width = width

        xx, yy = np.meshgrid(range(width), range(height))
        pts = np.stack([xx, yy, np.ones_like(xx)], axis=2)
        self.pts = nn.Parameter(torch.from_numpy(pts).float().unsqueeze(0).unsqueeze(4), requires_grad=False)

        self.pts_dict_eval = dict()

    def t2T(self, t, bz):
        T = torch.zeros([bz, 3, 3], device=t.device, dtype=t.dtype)
        T[:, 0, 1] = -t[:, 2]
        T[:, 0, 2] = t[:, 1]
        T[:, 1, 0] = t[:, 2]
        T[:, 1, 2] = -t[:, 0]
        T[:, 2, 0] = -t[:, 1]
        T[:, 2, 1] = t[:, 0]
        return T

    def epp_constrain(self, flowest, intrinsic, rel_pose, valid):
        bz, _, h, w = flowest.shape
        assert (self.height == h) and (self.width == w)

        flowest_ex = torch.cat([flowest, torch.zeros([bz, 1, h, w], device=flowest.device, dtype=flowest.dtype)], dim=1).permute([0, 2, 3, 1]).unsqueeze(4)

        ptse1 = self.pts.expand([bz, -1, -1, -1, -1])
        ptse2 = ptse1 + flowest_ex

        R = rel_pose[:, 0:3, 0:3]
        t = rel_pose[:, 0:3, 3] / torch.norm(rel_pose[:, 0:3, 3], dim=1, keepdim=True).expand([-1, 3])
        T = self.t2T(t, bz)
        F = T @ R
        E = torch.inverse(intrinsic).transpose(dim0=2, dim1=1) @ F @ torch.inverse(intrinsic)

        Ee = E.unsqueeze(1).unsqueeze(1).expand([-1, h, w, -1, -1])

        eppcons = torch.sum(ptse2.transpose(dim0=3, dim1=4) @ Ee * ptse1.transpose(dim0=3, dim1=4), dim=[3, 4])
        eppcons_loss = torch.sum(torch.abs(eppcons) * valid) / (torch.sum(valid) + 1)

        return eppcons_loss

    def epp_constrain_val(self, flowest, intrinsic, rel_pose, valid):
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

        R = rel_pose[:, 0:3, 0:3]
        t = rel_pose[:, 0:3, 3] / torch.norm(rel_pose[:, 0:3, 3], dim=1, keepdim=True).expand([-1, 3])
        T = self.t2T(t, bz)
        F = T @ R
        E = torch.inverse(intrinsic).transpose(dim0=2, dim1=1) @ F @ torch.inverse(intrinsic)

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

    def newton_gauss_F(self, pts2d1, pts2d2, intrinsic, posegt):
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

        loss_mv = 1 - torch.sum(t_est * t_gt)
        loss_ang = torch.mean((self.rot2ang((posegt[0:3, 0:3])) - self.rot2ang(R_est)).abs())

        print("\nOptimization finished at step %d, mv loss: %f, ang loss: %f, norm: %f, in all points: %f, gt pose: %f" % (
            kk, loss_mv, loss_ang, torch.norm(t_reg), loss_est, loss_gt))

        outputsrec['loss_mv'] = float(loss_mv.detach().cpu().numpy())
        outputsrec['loss_ang'] = float(loss_ang.detach().cpu().numpy())
        outputsrec['loss_constrain'] = float(loss_est.detach().cpu().numpy())
        outputsrec['loss_constrain_gt'] = float(loss_gt.detach().cpu().numpy())

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

        eppCbck = eppConstrainer_background(height=args.image_size[0], width=args.image_size[1])
        eppCbck.to(f'cuda:{args.gpu}')

        eppconcluer = eppConcluer()
        eppconcluer.to(f'cuda:{args.gpu}')
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

    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_entries, evaluation_entries = read_splits()

    aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
    train_dataset = KITTI_eigen(aug_params, split='training', root=args.dataset_root, entries=train_entries, semantics_root=args.semantics_root, depth_root=args.depth_root)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True,
                                   shuffle=(train_sampler is None), num_workers=int(args.num_workers / ngpus_per_node), drop_last=True,
                                   sampler=train_sampler)

    eval_dataset = KITTI_eigen(split='evaluation', root=args.dataset_root, entries=evaluation_entries, semantics_root=args.semantics_root, depth_root=args.depth_root)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset) if args.distributed else None
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=True,
                                   shuffle=(eval_sampler is None), num_workers=1, drop_last=True,
                                   sampler=eval_sampler)

    if args.distributed:
        group = dist.new_group([i for i in range(ngpus_per_node)])

    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)

    if args.gpu == 0:
        logger = Logger(model, scheduler, logroot, args.num_steps)
        logger_evaluation = Logger(model, scheduler, os.path.join(args.logroot, 'evaluation_eigen_background', args.name), args.num_steps)

    VAL_FREQ = 5000
    add_noise = False
    epoch = 0

    # print(validate_kitti(model.module, args, eval_loader, eppCbck, eppconcluer, group))
    should_keep_training = True
    while should_keep_training:

        train_sampler.set_epoch(epoch)
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()

            image1 = data_blob['img1']
            image1 = Variable(image1, requires_grad=True)
            image1 = image1.cuda(gpu, non_blocking=True)

            image2 = data_blob['img2']
            image2 = Variable(image2, requires_grad=True)
            image2 = image2.cuda(gpu, non_blocking=True)

            flow = data_blob['flow']
            flow = Variable(flow, requires_grad=True)
            flow = flow.cuda(gpu, non_blocking=True)

            valid = data_blob['valid']
            valid = Variable(valid, requires_grad=True)
            valid = valid.cuda(gpu, non_blocking=True)

            intrinsic = data_blob['intrinsic']
            intrinsic = Variable(intrinsic, requires_grad=True)
            intrinsic = intrinsic.cuda(gpu, non_blocking=True)

            rel_pose = data_blob['rel_pose']
            rel_pose = Variable(rel_pose, requires_grad=True)
            rel_pose = rel_pose.cuda(gpu, non_blocking=True)

            semantic_selector = data_blob['semantic_selector']
            semantic_selector = Variable(semantic_selector, requires_grad=True)
            semantic_selector = semantic_selector.cuda(gpu, non_blocking=True)

            if args.gpu == 0:
                print("current batch is %d" % i_batch)
            continue


            if add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda(gpu, non_blocking=True)).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda(gpu, non_blocking=True)).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)

            metrics = dict()
            loss_flow, metrics_flow = sequence_flowloss(flow_predictions, flow, valid, args.gamma)
            loss_eppc, metrics_eppc = sequence_eppcloss(eppCbck, flow_predictions, semantic_selector, intrinsic, rel_pose, args.gamma)

            metrics.update(metrics_flow)
            metrics.update(metrics_eppc)

            loss = loss_flow + loss_eppc * args.eppcw
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            if args.gpu == 0:
                logger.push(metrics, image1, image2, flow, flow_predictions, valid, data_blob['depth'])

            if total_steps % VAL_FREQ == VAL_FREQ - 1:

                results = validate_kitti(model.module, args, eval_loader, eppCbck, eppconcluer, group)

                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()

                if args.gpu == 0:
                    logger_evaluation.write_dict(results, total_steps)
                    PATH = os.path.join(logroot, '%s.pth' % (str(total_steps + 1).zfill(3)))
                    torch.save(model.state_dict(), PATH)

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break
        epoch = epoch + 1

    if args.gpu == 0:
        logger.close()
        PATH = os.path.join(logroot, 'final.pth')
        torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

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