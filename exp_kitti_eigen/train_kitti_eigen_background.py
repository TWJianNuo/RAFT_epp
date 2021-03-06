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
def validate_kitti(model, args, eval_loader, eppCbck, group, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    epe_list = torch.zeros(2).cuda(device=args.gpu)
    out_list = torch.zeros(2).cuda(device=args.gpu)
    eppc_list = torch.zeros(2).cuda(device=args.gpu)

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

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0])

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

    if args.distributed:
        dist.all_reduce(tensor=epe_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=out_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=eppc_list, op=dist.ReduceOp.SUM, group=group)

    if args.gpu == 0:
        epe = epe_list[0] / epe_list[1]
        f1 = 100 * out_list[0] / out_list[1]
        eppc = eppc_list[0] / eppc_list[1]

        print("Validation KITTI, epe: %f, f1: %f, eppc: %f" % (epe, f1, eppc))
        return {'kitti-epe': float(epe.detach().cpu().numpy()), 'kitti-f1': float(f1.detach().cpu().numpy()), 'kitti-eppc': float(eppc.detach().cpu().numpy())}
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
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=False,
                                   shuffle=(train_sampler is None), num_workers=args.num_workers, drop_last=True,
                                   sampler=train_sampler)

    eval_dataset = KITTI_eigen(split='evaluation', root=args.dataset_root, entries=evaluation_entries, semantics_root=args.semantics_root, depth_root=args.depth_root)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset) if args.distributed else None
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=False,
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

                results = validate_kitti(model.module, args, eval_loader, eppCbck, group)

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
    parser.add_argument('--dist_url', type=str, help='url used to set up distributed training', default='tcp://127.0.0.1:1235')
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