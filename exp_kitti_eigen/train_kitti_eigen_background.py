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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from exp_kitti_eigen.dataset_kitti_eigen import KITTI_eigen

from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import PIL.Image as Image
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
# SUM_FREQ = 100
SUM_FREQ = 10
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

def sequence_epploss(flow_preds, flow_gt, valid, semantic_selector, intrinsic, rel_pose, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    T, R = pose2RT(rel_pose)
    F = T @ R
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        curflow = flow_preds[i]

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


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

    def push(self, metrics, image1, image2, flowgt, flow_predictions):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.write_vls(image1, image2, flowgt, flow_predictions)
            self.running_loss = {}

    def write_vls(self, image1, image2, flowgt, flow_predictions):
        img1 = image1[0].cpu().detach().permute([1, 2, 0]).numpy().astype(np.uint8)
        img2 = image2[0].cpu().detach().permute([1, 2, 0]).numpy().astype(np.uint8)
        flow_pred = flow_to_image(flow_predictions[-1][0].permute([1, 2, 0]).detach().cpu().numpy())
        flow_gt = flow_to_image(flowgt[0].permute([1, 2, 0]).detach().cpu().numpy())

        h, w = image1.shape[2::]
        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        pixelloc = np.stack([xx, yy], axis=0)
        pixelloc = torch.from_numpy(pixelloc).float() + flow_predictions[-1][0].detach().cpu()
        pixelloc[0, :, :] = ((pixelloc[0, :, :] / w) - 0.5) * 2
        pixelloc[1, :, :] = ((pixelloc[1, :, :] / h) - 0.5) * 2
        pixelloc = pixelloc.permute([1, 2, 0])

        image1_recon = torch.nn.functional.grid_sample(image2[0, :, :, :].detach().unsqueeze(0).cpu(), pixelloc.unsqueeze(0), mode='bilinear', align_corners=False)
        image1_recon = image1_recon[0].permute([1, 2, 0]).numpy().astype(np.uint8)

        img_up = np.concatenate([img1, img2], axis=1)
        img_mid = np.concatenate([flow_pred, flow_gt], axis=1)
        img_down = np.concatenate([img1, image1_recon], axis=1)
        img_vls = np.concatenate([img_up, img_mid, img_down], axis=0)

        self.writer.add_image('img_vls', (torch.from_numpy(img_vls).float() / 255).permute([2, 0, 1]), self.total_steps)

    def write_dict(self, results, step):
        self.create_summarywriter()

        for key in results:
            self.writer.add_scalar(key, results[key], step)

    def close(self):
        self.writer.close()

@torch.no_grad()
def validate_kitti(model, args, eval_loader, group, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    epe_list = torch.zeros(2).cuda(device=args.gpu)
    out_list = torch.zeros(2).cuda(device=args.gpu)

    for val_id, batch in enumerate(tqdm(eval_loader)):
        image1 = batch['img1']
        image1 = Variable(image1, requires_grad=True)
        image1 = image1.cuda(args.gpu, non_blocking=True)

        image2 = batch['img2']
        image2 = Variable(image2, requires_grad=True)
        image2 = image2.cuda(args.gpu, non_blocking=True)

        flow_gt = batch['flow']
        flow_gt = Variable(flow_gt, requires_grad=True)
        flow_gt = flow_gt.cuda(args.gpu, non_blocking=True)
        flow_gt = flow_gt[0]

        valid_gt = batch['valid']
        valid_gt = Variable(valid_gt, requires_grad=True)
        valid_gt = valid_gt.cuda(args.gpu, non_blocking=True)
        valid_gt = valid_gt[0]

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0])

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

    if args.distributed:
        dist.all_reduce(tensor=epe_list, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=out_list, op=dist.ReduceOp.SUM, group=group)

    if args.gpu == 0:
        epe = epe_list[0] / epe_list[1]
        f1 = 100 * out_list[0] / out_list[1]

        print("Validation KITTI: %f, %f" % (epe, f1))
        return {'kitti-epe': float(epe.detach().cpu().numpy()), 'kitti-f1': float(f1.detach().cpu().numpy())}
    else:
        return None

def read_splits():
    split_root = os.path.join(project_rootdir, 'exp_kitti_eigen', 'splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'train_files.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'test_files.txt'), 'r')]
    return train_entries, evaluation_entries

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
                                   shuffle=(eval_sampler is None), num_workers=args.num_workers, drop_last=True,
                                   sampler=eval_sampler)

    if args.distributed:
        group = dist.new_group([i for i in range(ngpus_per_node)])

    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)

    if args.gpu == 0:
        logger = Logger(model, scheduler, logroot)
        logger_evaluation = Logger(model, scheduler, os.path.join(args.logroot, 'evaluation', args.name))

    VAL_FREQ = 5000
    add_noise = True
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

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda(gpu, non_blocking=True)).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda(gpu, non_blocking=True)).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)

            metrics = dict()
            loss_flow, metrics_flow = sequence_flowloss(flow_predictions, flow, valid, args.gamma)
            loss_epp, metrics_epp = sequence_epploss(flow_predictions, flow, valid, semantic_selector, intrinsic, rel_pose, args.gamma)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            if args.gpu == 0:
                logger.push(metrics, image1, image2, flow, flow_predictions)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:

                results = validate_kitti(model.module, args, eval_loader, group)

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