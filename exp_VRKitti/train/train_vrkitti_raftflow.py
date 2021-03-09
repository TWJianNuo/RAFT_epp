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
from core.raft import RAFT
import copy

from exp_VRKitti.dataset_VRKitti2 import VirtualKITTI2

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
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
        self.running_loss = {}
        self.logpath = logpath
        self.writer = None

    def create_summarywriter(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.logpath)

    def write_vls(self, data_blob, flowpred, selector, step):
        img1 = data_blob['img1'][0].permute([1, 2, 0]).numpy().astype(np.uint8)
        insmap = data_blob['insmap'][0].squeeze().numpy()

        figmask = tensor2disp(selector, vmax=1, viewind=0)
        insvls = Image.fromarray(vls_ins(img1, insmap))

        flowgtvls = Image.fromarray(flow_to_image(data_blob['flowmap'][0].permute([1, 2, 0]).numpy(), rad_max=10))
        flowpredvls = Image.fromarray(flow_to_image(flowpred[-1][0].detach().cpu().permute([1, 2, 0]).numpy(), rad_max=10))

        img_val_up = np.concatenate([np.array(figmask), np.array(insvls)], axis=1)
        img_val_down = np.concatenate([np.array(flowgtvls), np.array(flowpredvls)], axis=1)
        img_val = np.concatenate([np.array(img_val_up), np.array(img_val_down)], axis=0)
        self.writer.add_image('predvls', (torch.from_numpy(img_val).float() / 255).permute([2, 0, 1]), step)

    def write_vls_eval(self, data_blob, depth, selector, evalidx, step):
        img1 = data_blob['img1'][0].permute([1, 2, 0]).numpy().astype(np.uint8)
        insmap = data_blob['insmap'][0].squeeze().numpy()

        figmask = tensor2disp(selector, vmax=1, viewind=0)
        insvls = Image.fromarray(vls_ins(img1, insmap))

        depthgtvls = tensor2disp(1 / data_blob['depthmap'], vmax=0.15, viewind=0)
        depthpredvls = tensor2disp(1 / depth, vmax=0.15, viewind=0)

        img_val_up = np.concatenate([np.array(figmask), np.array(insvls)], axis=1)
        img_val_down = np.concatenate([np.array(depthgtvls), np.array(depthpredvls)], axis=1)
        img_val = np.concatenate([np.array(img_val_up), np.array(img_val_down)], axis=0)
        self.writer.add_image('predvls_eval_{}'.format(str(evalidx).zfill(2)), (torch.from_numpy(img_val).float() / 255).permute([2, 0, 1]), step)

    def write_dict(self, results, step):
        for key in results:
            self.writer.add_scalar(key, results[key], step)

    def close(self):
        self.writer.close()

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

@torch.no_grad()
def validate_VRKitti2(model, args, eval_loader, iters, group, logger, total_steps):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    gpu = args.gpu
    eval_epe = torch.zeros(2).cuda(device=gpu)
    eval_out = torch.zeros(2).cuda(device=gpu)

    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        image1 = data_blob['img1'].cuda(gpu, non_blocking=True)
        image2 = data_blob['img2'].cuda(gpu, non_blocking=True)
        flow_gt = data_blob['flowmap'].cuda(gpu, non_blocking=True)

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        valid = ((flow_gt[:, 0] != 0) * (flow_gt[:, 1] != 0) * (mag < MAX_FLOW)).unsqueeze(1).float()
        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)

        epe = torch.sum((flow_pr - flow_gt)**2, dim=1, keepdim=True).sqrt()
        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()

        out = ((epe > 3.0) * ((epe / (mag + 1e-10)) > 0.05) * valid).float()

        eval_out[0] += torch.sum(out)
        eval_out[1] += torch.sum(valid)

        eval_epe[0] += torch.sum(torch.sum(epe * valid, dim=[1, 2, 3]) / torch.sum(valid, dim=[1, 2, 3]))
        eval_epe[1] += image1.shape[0]

        if not(logger is None) and np.mod(val_id, 40) == 0:
            logger.write_vls_eval(data_blob, flow_pr, valid, val_id, total_steps)

    if args.distributed:
        dist.all_reduce(tensor=eval_out, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=eval_epe, op=dist.ReduceOp.SUM, group=group)

    if args.gpu == 0:
        eval_out[0] = eval_out[0] / eval_out[1]
        eval_epe[0] = eval_epe[0] / eval_epe[1]

        if iters == 24:
            print("in {} eval samples: Out: {:7.3f}, Epe: {:7.3f}".format(eval_epe[1].item(), eval_out[0].item(), eval_epe[0].item()))

        return {'out': float(eval_out[0].item()), 'epe': float(eval_epe[0].item())}
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

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

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

    train_entries, evaluation_entries = read_splits()
    train_dataset = VirtualKITTI2(args=args, root=args.dataset_root, inheight=args.inheight, inwidth=args.inwidth, entries=train_entries, istrain=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=False,
                                   shuffle=(train_sampler is None), num_workers=args.num_workers, drop_last=True,
                                   sampler=train_sampler)

    eval_dataset = VirtualKITTI2(args=args, root=args.dataset_root, inheight=args.evalheight, inwidth=args.evalwidth, entries=evaluation_entries, istrain=False)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset) if args.distributed else None
    eval_loader = data.DataLoader(eval_dataset, batch_size=args.batch_size, pin_memory=False,
                                   shuffle=(eval_sampler is None), num_workers=0, drop_last=True,
                                   sampler=eval_sampler)

    print("Training split contains %d images, validation split contained %d images" % (len(train_entries), len(evaluation_entries)))

    if args.distributed:
        group = dist.new_group([i for i in range(ngpus_per_node)])

    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    VAL_ITERINC = 4

    if args.gpu == 0:
        logger = Logger(model, scheduler, logroot)
        logger.create_summarywriter()

        logger_evaluations = dict()
        for num_iters in range(args.iters, args.iters * 2 + 1, VAL_ITERINC):
            logger_evaluation = Logger(model, scheduler, os.path.join(args.logroot, 'evaluation_VRKitti', "{}_iternum{}".format(args.name, str(num_iters).zfill(2))))
            logger_evaluation.create_summarywriter()
            logger_evaluations[num_iters] = logger_evaluation

    VAL_FREQ = 500
    maxout = 1
    epoch = 0

    st = time.time()
    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()

            image1 = data_blob['img1'].cuda(gpu, non_blocking=True)
            image2 = data_blob['img2'].cuda(gpu, non_blocking=True)
            flow = data_blob['flowmap'].cuda(gpu, non_blocking=True)

            # exlude invalid pixels and extremely large diplacements
            mag = torch.sum(flow ** 2, dim=1).sqrt()
            valid = ((flow[:, 0] > 0) * (flow[:, 1] > 0) * (mag < MAX_FLOW)).unsqueeze(1)

            flow_predictions = model(image1, image2, iters=args.iters)
            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)

            metrics = dict()
            metrics['loss_flow'] = loss.float().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            if args.gpu == 0:
                logger.write_dict(metrics, total_steps)
                if total_steps % SUM_FREQ == 0:
                    dr = time.time() - st
                    resths = (args.num_steps - total_steps) * dr / (total_steps + 1) / 60 / 60
                    print("Step: %d, rest hour: %f, flow loss: %f" % (total_steps, resths, loss.item()))
                    logger.write_vls(data_blob, flow_predictions, valid, total_steps)

            if total_steps % VAL_FREQ == 1:
                for num_iters in range(args.iters, args.iters * 2 + 1, VAL_ITERINC):

                    if args.gpu == 0 and num_iters == 24:
                        results = validate_VRKitti2(model.module, args, eval_loader, args.iters, group, logger, total_steps)
                    else:
                        results = validate_VRKitti2(model.module, args, eval_loader, args.iters, group, None, None)

                    if args.gpu == 0:
                        logger_evaluations[num_iters].write_dict(results, total_steps)

                        if num_iters == 24:
                            if results['out'] < maxout:
                                maxout = results['out']
                                PATH = os.path.join(logroot, 'minout.pth')
                                torch.save(model.state_dict(), PATH)
                                print("model saved to %s" % PATH)

                model.train()
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
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--inheight', type=int, default=288)
    parser.add_argument('--inwidth', type=int, default=576)
    parser.add_argument('--evalheight', type=int, default=288)
    parser.add_argument('--evalwidth', type=int, default=960)

    parser.add_argument('--maxinsnum', type=int, default=20)
    parser.add_argument('--tscale_range', type=float, default=3)
    parser.add_argument('--objtscale_range', type=float, default=10)
    parser.add_argument('--angx_range', type=float, default=0.03)
    parser.add_argument('--angy_range', type=float, default=0.06)
    parser.add_argument('--angz_range', type=float, default=0.01)

    parser.add_argument('--num_deges', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=50)
    parser.add_argument('--min_depth_pred', type=float, default=1)
    parser.add_argument('--max_depth_pred', type=float, default=85)
    parser.add_argument('--min_depth_eval', type=float, default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, default=80)

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
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