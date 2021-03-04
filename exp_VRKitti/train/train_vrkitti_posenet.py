from __future__ import print_function, division
import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import PIL.Image as Image
from core.utils.flow_viz import flow_to_image
from core.utils.utils import InputPadder, forward_interpolate
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable
from eppcore import eppcore_inflation, eppcore_compression
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
from core.utils.utils import tensor2disp, tensor2rgb
from posenet import Posenet
import copy

from exp_VRKitti.dataset_VRKitti2 import VirtualKITTI2

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 500
VAL_FREQ = 5000

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler

def vls_ins(rgb, anno):
    rgbc = copy.deepcopy(rgb)
    r = rgbc[:, :, 0].astype(np.float)
    g = rgbc[:, :, 1].astype(np.float)
    b = rgbc[:, :, 2].astype(np.float)
    for i in np.unique(anno):
        if i > 0:
            rndc = np.random.randint(0, 255, 3).astype(np.float)
            selector = anno == i
            r[selector] = rndc[0] * 0.25 + r[selector] * 0.75
            g[selector] = rndc[1] * 0.25 + g[selector] * 0.75
            b[selector] = rndc[2] * 0.25 + b[selector] * 0.75
    rgbvls = np.stack([r, g, b], axis=2)
    rgbvls = np.clip(rgbvls, a_max=255, a_min=0).astype(np.uint8)
    return rgbvls

class Logger:
    def __init__(self, model, scheduler, logpath):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.logpath = logpath
        self.writer = None

    def create_summarywriter(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.logpath)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        self.create_summarywriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics, data_blob, est_objpose, estflow, selector):
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]
        if self.total_steps % SUM_FREQ == 0:
            self._print_training_status()
            self.write_vls(data_blob, est_objpose, estflow, selector)
            self.running_loss = {}

        self.total_steps += 1

    def write_vls(self, data_blob, est_objpose, estflow, selector):
        img1 = data_blob['img1'][0].permute([1, 2, 0]).numpy().astype(np.uint8)
        img2 = data_blob['img2'][0].permute([1, 2, 0]).numpy().astype(np.uint8)

        depthmap = data_blob['depthmap'][0].squeeze().numpy()
        insmap = data_blob['insmap'][0].squeeze().numpy()
        intrinsic = data_blob['intrinsic'][0].squeeze().numpy()
        est_objpose = est_objpose[0].detach().cpu().numpy()
        flowmap = data_blob['flowmap'][0].squeeze().numpy()
        objpose_gt = data_blob['poses'][0].squeeze().numpy()

        figmask = tensor2disp(selector, vmax=1, viewind=0)
        insvls = Image.fromarray(vls_ins(img1, insmap))

        posevls = self.plot_scattervls(img1, img2, depthmap, insmap, intrinsic, objpose_gt[0], objpose_gt, est_objpose, flowmap)
        flowvls = flow_to_image(data_blob['flowmap'][0].cpu().permute([1,2,0]).numpy(), rad_max=15)
        estflowvls = flow_to_image(estflow[0].cpu().permute([1,2,0]).numpy(), rad_max=15)

        img_val_up = np.concatenate([np.array(figmask), np.array(insvls)], axis=1)
        img_val_down = np.concatenate([np.array(flowvls), np.array(estflowvls)], axis=1)
        img_val = np.concatenate([np.array(img_val_up), np.array(img_val_down)], axis=0)
        posevls = np.array(posevls[:, :, 0:3])
        self.writer.add_image('img_val', (torch.from_numpy(img_val).float() / 255).permute([2, 0, 1]), self.total_steps)
        self.writer.add_image('posevls', (torch.from_numpy(posevls).float() / 255).permute([2, 0, 1]), self.total_steps)

        # Image.fromarray(flow_to_image(flow[0].cpu().permute([1,2,0]).numpy(), rad_max=10)).show()
        # Image.fromarray(flow_to_image(flowpred[0].cpu().permute([1,2,0]).numpy(), rad_max=10)).show()
        # tensor2rgb(image1, viewind=0).show()
        # tensor2disp(1 / depthmap, vmax=0.15, viewind=0).show()

    def plot_scattervls(self, img1, img2, depthmap, instancemap, intrinsic, relpose, obj_poses, obj_approxposes, flowmap):
        h, w = depthmap.shape
        cm = plt.get_cmap('magma')
        vmax = 0.15

        staticsel = instancemap == 0
        samplenume = 10000
        rndx = np.random.randint(0, w, [samplenume])
        rndy = np.random.randint(0, h, [samplenume])
        rndsel = staticsel[rndy, rndx]

        rndx = rndx[rndsel]
        rndy = rndy[rndsel]
        rndd = depthmap[rndy, rndx]

        intrinsic44 = np.eye(4)
        intrinsic44[0:3, 0:3] = intrinsic

        rndpts = np.stack([rndx * rndd, rndy * rndd, rndd, np.ones_like(rndd)], axis=0)
        rndpts = intrinsic44 @ relpose @ np.linalg.inv(intrinsic44) @ rndpts
        rndptsx = rndpts[0, :] / rndpts[2, :]
        rndptsy = rndpts[1, :] / rndpts[2, :]

        rndflowx = flowmap[0, rndy, rndx]
        rndflowy = flowmap[1, rndy, rndx]

        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        objsample = 1000
        orgpts = list()
        flowpts = list()
        objposepts = list()
        objposepts_approx = list()
        colors = list()
        for k in np.unique(instancemap):
            if k == 0:
                continue
            obj_pose = obj_poses[k]
            obj_pose_approx = obj_approxposes[k]
            obj_sel = instancemap == k
            if np.sum(obj_sel) == 0:
                continue

            rndidx = np.random.randint(0, np.sum(obj_sel), objsample)

            objxxf = xx[obj_sel][rndidx]
            objyyf = yy[obj_sel][rndidx]

            objxxff = flowmap[0, objyyf, objxxf]
            objyyff = flowmap[1, objyyf, objxxf]
            objdf = depthmap[objyyf, objxxf]
            objcolor = 1 / objdf / vmax
            objcolor = cm(objcolor)

            objpts3d = np.stack([objxxf * objdf, objyyf * objdf, objdf, np.ones_like(objdf)], axis=0)
            objpts3d = intrinsic44 @ obj_pose @ np.linalg.inv(intrinsic44) @ objpts3d
            objpts3dx = objpts3d[0, :] / objpts3d[2, :]
            objpts3dy = objpts3d[1, :] / objpts3d[2, :]

            objpts3d = np.stack([objxxf * objdf, objyyf * objdf, objdf, np.ones_like(objdf)], axis=0)
            objpts3d = intrinsic44 @ obj_pose_approx @ np.linalg.inv(intrinsic44) @ objpts3d
            objpts3dx_approx = objpts3d[0, :] / objpts3d[2, :]
            objpts3dy_approx = objpts3d[1, :] / objpts3d[2, :]

            objxxf_o = objxxf + objxxff
            objyyf_o = objyyf + objyyff

            orgpts.append(np.stack([objxxf, objyyf], axis=0))
            flowpts.append(np.stack([objxxf_o, objyyf_o], axis=0))
            objposepts.append(np.stack([objpts3dx, objpts3dy], axis=0))
            objposepts_approx.append(np.stack([objpts3dx_approx, objpts3dy_approx], axis=0))
            colors.append(objcolor)

        tnp = 1 / rndd / vmax
        tnp = cm(tnp)

        fig = plt.figure(figsize=(16, 9))
        canvas = FigureCanvasAgg(fig)
        fig.add_subplot(2, 2, 1)
        plt.scatter(rndx, rndy, 1, tnp)
        for k in range(len(orgpts)):
            plt.scatter(orgpts[k][0], orgpts[k][1], 1, colors[k])
        plt.imshow(img1)
        plt.title('Current frame')

        fig.add_subplot(2, 2, 2)
        plt.scatter(rndx + rndflowx, rndy + rndflowy, 1, tnp)
        for k in range(len(flowpts)):
            plt.scatter(flowpts[k][0], flowpts[k][1], 1, colors[k])
        plt.imshow(img2)
        plt.title('flow gt')

        fig.add_subplot(2, 2, 3)
        plt.scatter(rndptsx, rndptsy, 1, tnp)
        for k in range(len(objposepts)):
            plt.scatter(objposepts[k][0], objposepts[k][1], 1, colors[k])
        plt.imshow(img2)
        plt.title('pose gt')

        fig.add_subplot(2, 2, 4)
        plt.scatter(rndptsx, rndptsy, 1, tnp)
        for k in range(len(objposepts_approx)):
            plt.scatter(objposepts_approx[k][0], objposepts_approx[k][1], 1, colors[k])
        plt.imshow(img2)
        plt.title('pose est')

        fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
        canvas.draw()
        buf = canvas.buffer_rgba()
        plt.close()
        X = np.asarray(buf)
        X = np.array(Image.fromarray(X).resize([1600, 900], Image.BILINEAR))

        return X

    def write_dict(self, results, step):
        self.create_summarywriter()

        for key in results:
            self.writer.add_scalar(key, results[key], step)

    def close(self):
        self.writer.close()

@torch.no_grad()
def validate_VRKitti2(model, args, eval_loader, group):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    epe_list = torch.zeros(2).cuda(device=args.gpu)
    out_list = torch.zeros(2).cuda(device=args.gpu)

    selfscale_list = torch.zeros(2).cuda(device=args.gpu)
    selftdir_list = torch.zeros(2).cuda(device=args.gpu)
    selfang_list = torch.zeros(2).cuda(device=args.gpu)
    objang_list = torch.zeros(2).cuda(device=args.gpu)
    objscale_list = torch.zeros(2).cuda(device=args.gpu)

    gpu = args.gpu
    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        image1 = data_blob['img1'].cuda(gpu, non_blocking=True)
        image2 = data_blob['img2'].cuda(gpu, non_blocking=True)
        flowgt = data_blob['flowmap'].cuda(gpu, non_blocking=True)
        depthmap = data_blob['depthmap'].cuda(gpu, non_blocking=True)
        intrinsic = data_blob['intrinsic'].cuda(gpu, non_blocking=True)
        insmap = data_blob['insmap'].cuda(gpu, non_blocking=True)
        poses = data_blob['poses'].cuda(gpu, non_blocking=True)
        ang = data_blob['ang'].cuda(gpu, non_blocking=True)
        scale = data_blob['scale'].cuda(gpu, non_blocking=True)

        t_gt = poses[:, 0, 0:3, 3]
        t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
        t_gt_nromed = t_gt / (t_gt_norm + 1e-10)
        R_gt = poses[:, 0, 0:3, 0:3]
        ang_gt = R2ang(R_gt)

        insnum = model.eppcompress(insmap, (insmap > -1).float().squeeze(1).unsqueeze(-1).unsqueeze(-1), args.maxinsnum)

        self_ang, self_tdir, self_tscale, obj_pose = model(image1, image2)
        selfR, selfT, selfRT = model.get_selfpose(selfang=self_ang, selftdir=self_tdir, selfscale=self_tscale)

        objscale_pred = model.eppcompress(insmap, obj_pose[('obj_scale', 0)].squeeze(1).unsqueeze(-1).unsqueeze(-1), args.maxinsnum)
        objscale_pred = objscale_pred / (insnum + 1e-10)

        objang_pred = model.eppcompress(insmap, obj_pose[('obj_angle', 0)].squeeze(1).unsqueeze(-1).unsqueeze(-1), args.maxinsnum)
        objang_pred = objang_pred / (insnum + 1e-10)

        est_objpose = model.mvinfo2objpose(objang_pred, objscale_pred, selfRT.unsqueeze(1).expand([-1, args.maxinsnum, -1, -1]))
        est_allpose = torch.clone(est_objpose)
        est_allpose[:, 0, :, :] = selfRT

        flowpred = model.depth2flow(depthmap=depthmap, instance=insmap, intrinsic=intrinsic, t=est_allpose[:, :, 0:3, 3:4], R=est_allpose[:, :, 0:3, 0:3])

        if torch.sum(insnum > 0) > 1:
            objang_list[0] += (objang_pred[:, 1::, :, :] - ang[:, 1::, :, :]).abs().mean().item()
            objang_list[1] += 1

            objscale_list[0] += (objscale_pred[:, 1::, :, :] - scale[:, 1::, :, :]).abs().mean().item()
            objscale_list[1] += 1

        selfscale_list[0] += (t_gt_norm - self_tscale).abs().mean().item()
        selfscale_list[1] += 1

        selftdir_list[0] += (t_gt_nromed - self_tdir).abs().mean().item()
        selftdir_list[1] += 1

        selfang_list[0] += (ang_gt - self_ang).abs().mean().item()
        selfang_list[1] += 1

        selector = (insmap >= 0).float().expand([-1, 2, -1, -1])

        epe = torch.sum(((flowpred - flowgt) * selector)**2, dim=0).sqrt()
        mag = torch.sum((flowgt * selector)**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()

        epe_list[0] += epe.mean().item()
        epe_list[1] += 1

        out_list[0] += out.sum()
        out_list[1] += torch.sum(selector)

    if args.distributed:
        dist.all_reduce(tensor=selfscale_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=selftdir_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=selfang_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=objang_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=objscale_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=epe_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=out_list, op=dist.ReduceOp.SUM, group=group)

    if args.gpu == 0:
        epe = epe_list[0] / epe_list[1]
        f1 = 100 * out_list[0] / out_list[1]
        loss_selfscale = selfscale_list[0] / selfscale_list[1]
        loss_selftdir = selftdir_list[0] / selftdir_list[1]
        loss_selfang = selfang_list[0] / selfang_list[1]
        loss_objang = objang_list[0] / objang_list[1]
        loss_objscale = objscale_list[0] / objscale_list[1]

        print("Validation performance: epe: %f, f1: %f, selfscale: %.3E, selftdir: %.3E, selfang: %.3E, objang: %.3E, objscale: %.3E" % (epe, f1, loss_selfscale, loss_selftdir, loss_selfang, loss_objang, loss_objscale))
        return {'kitti-epe': float(epe.detach().cpu().numpy()),
                'kitti-f1': float(f1.detach().cpu().numpy()),
                'selfscale': float(loss_selfscale.detach().cpu().numpy()),
                'selfdir': float(loss_selftdir.detach().cpu().numpy()),
                'selfang': float(loss_selfang.detach().cpu().numpy()),
                'objang': float(loss_objang.detach().cpu().numpy()),
                'objscale': float(loss_objscale.detach().cpu().numpy()),
                }
    else:
        return None

def read_splits():
    split_root = os.path.join(project_rootdir, 'exp_VRKitti', 'splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'training_split.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'evaluation_split.txt'), 'r')]
    return train_entries, evaluation_entries

def R2ang(R):
    # This is not an efficient implementation
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    ang0 = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    ang1 = torch.atan2(-R[:, 2, 0], sy)
    ang2 = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    ang = torch.stack([ang0, ang1, ang2], dim=1)
    return ang

def train(gpu, ngpus_per_node, args):
    print("Using GPU %d for training" % gpu)
    args.gpu = gpu

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=ngpus_per_node, rank=args.gpu)

    model = Posenet(args.num_layers, args)
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

    model.train()

    train_entries, evaluation_entries = read_splits()
    train_dataset = VirtualKITTI2(args=args, root=args.dataset_root, entries=train_entries)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=False,
                                   shuffle=(train_sampler is None), num_workers=args.num_workers, drop_last=True,
                                   sampler=train_sampler)

    eval_dataset = VirtualKITTI2(args=args, root=args.dataset_root, entries=evaluation_entries)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset) if args.distributed else None
    eval_loader = data.DataLoader(eval_dataset, batch_size=args.batch_size, pin_memory=False,
                                   shuffle=(eval_sampler is None), num_workers=args.num_workers, drop_last=True,
                                   sampler=eval_sampler)

    if args.distributed:
        group = dist.new_group([i for i in range(ngpus_per_node)])

    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0

    if args.gpu == 0:
        logger = Logger(model, scheduler, logroot)
        logger_evaluation = Logger(model, scheduler, os.path.join(args.logroot, 'evaluation_VRKitti', args.name))

    VAL_FREQ = 1000
    minf1 = 100
    epoch = 0

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()

            image1 = data_blob['img1'].cuda(gpu, non_blocking=True)
            image2 = data_blob['img2'].cuda(gpu, non_blocking=True)
            flow = data_blob['flowmap'].cuda(gpu, non_blocking=True)
            depthmap = data_blob['depthmap'].cuda(gpu, non_blocking=True)
            intrinsic = data_blob['intrinsic'].cuda(gpu, non_blocking=True)
            insmap = data_blob['insmap'].cuda(gpu, non_blocking=True)
            poses = data_blob['poses'].cuda(gpu, non_blocking=True)
            ang = data_blob['ang'].cuda(gpu, non_blocking=True)
            scale = data_blob['scale'].cuda(gpu, non_blocking=True)

            t_gt = poses[:, 0, 0:3, 3]
            t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
            t_gt_nromed = t_gt / (t_gt_norm + 1e-10)
            R_gt = poses[:, 0, 0:3, 0:3]
            ang_gt = R2ang(R_gt)

            angobj_inf = model.module.eppinflate(insmap, ang).squeeze(-1).squeeze(-1).unsqueeze(1)
            scaleobj_inf = model.module.eppinflate(insmap, scale).squeeze(-1).squeeze(-1).unsqueeze(1)
            insnum = model.module.eppcompress(insmap, (insmap > -1).float().squeeze(1).unsqueeze(-1).unsqueeze(-1), args.maxinsnum)

            self_ang, self_tdir, self_tscale, obj_pose = model(image1, image2)

            loss_self_ang = torch.abs(self_ang - ang_gt).mean()
            loss_self_tdir = torch.abs(self_tdir - t_gt_nromed).mean()
            loss_self_tscale = torch.abs(self_tscale - t_gt_norm).mean()

            selector = (insmap > 0).float()
            loss_objscale = 0
            loss_objang = 0
            for k in range(4):
                objscale_pred = F.interpolate(obj_pose[('obj_scale', k)], [args.inheight, args.inwidth], mode='bilinear', align_corners=False)
                objang_pred = F.interpolate(obj_pose[('obj_angle', k)], [args.inheight, args.inwidth], mode='bilinear', align_corners=False)
                loss_objscale += torch.sum(torch.abs(objscale_pred - scaleobj_inf) * selector) / (torch.sum(selector) + 1)
                loss_objang += torch.sum(torch.abs(objang_pred - angobj_inf) * selector) / (torch.sum(selector) + 1)
            loss_objscale = loss_objscale / 4
            loss_objang = loss_objang / 4

            loss = loss_self_ang + loss_self_tdir + loss_self_tscale + loss_objscale + loss_objang

            metrics = dict()
            metrics['loss_self_ang'] = loss_self_ang.float().item()
            metrics['loss_self_tdir'] = loss_self_tdir.float().item()
            metrics['loss_self_tscale'] = loss_self_tscale.float().item()
            metrics['loss_objscale'] = loss_objscale.float().item()
            metrics['loss_objang'] = loss_objang.float().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            if args.gpu == 0:
                if total_steps % SUM_FREQ == 0:
                    with torch.no_grad():
                        objscale_pred = model.module.eppcompress(insmap, obj_pose[('obj_scale', 0)].squeeze(1).unsqueeze(-1).unsqueeze(-1), args.maxinsnum)
                        objscale_pred = objscale_pred / (insnum + 1e-10)

                        objang_pred = model.module.eppcompress(insmap, obj_pose[('obj_angle', 0)].squeeze(1).unsqueeze(-1).unsqueeze(-1), args.maxinsnum)
                        objang_pred = objang_pred / (insnum + 1e-10)

                        selfR, selfT, selfRT = model.module.get_selfpose(selfang=self_ang, selftdir=self_tdir, selfscale=self_tscale)
                        est_objpose = model.module.mvinfo2objpose(objang_pred, objscale_pred, selfRT.unsqueeze(1).expand([-1, args.maxinsnum, -1, -1]))

                        est_allpose = torch.clone(est_objpose)
                        est_allpose[:, 0, :, :] = selfRT

                        flowpred = model.module.depth2flow(depthmap=depthmap, instance=insmap, intrinsic=intrinsic, t=est_allpose[:, :, 0:3, 3:4], R=est_allpose[:, :, 0:3, 0:3])

                    logger.push(metrics, data_blob, est_objpose, flowpred, selector)
                else:
                    logger.push(metrics, None, None, None, None)

            if total_steps % VAL_FREQ == 1:
                results = validate_VRKitti2(model.module, args, eval_loader, group)

                model.train()
                if args.gpu == 0:
                    logger_evaluation.write_dict(results, total_steps)

                    if results['kitti-f1'] < minf1:
                        minf1 = results['kitti-f1']
                        PATH = os.path.join(logroot, 'minf1.pth')
                        torch.save(model.state_dict(), PATH)
                        print("model saved to %s" % PATH)

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
    parser.add_argument('--restore_ckpt', help="restore checkpoint")

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--inheight', type=int, default=288)
    parser.add_argument('--inwidth', type=int, default=960)
    parser.add_argument('--maxinsnum', type=int, default=20)
    parser.add_argument('--maxscale', type=float, default=10)

    parser.add_argument('--tscale_range', type=float, default=3)
    parser.add_argument('--objtscale_range', type=float, default=10)
    parser.add_argument('--angx_range', type=float, default=0.03)
    parser.add_argument('--angy_range', type=float, default=0.06)
    parser.add_argument('--angz_range', type=float, default=0.01)
    parser.add_argument('--num_layers', type=int, default=50)

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--logroot', type=str)
    parser.add_argument('--num_workers', type=int, default=12)

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