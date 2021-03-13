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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader
from exp_kitti_eigen_fixation.dataset_kitti_eigen_fixation import KITTI_eigen
from exp_kitti_eigen_fixation.eppflowenet.EppFlowNet import EppFlowNet

from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from PIL import Image, ImageDraw
from core.utils.flow_viz import flow_to_image
from core.utils.utils import InputPadder, forward_interpolate, tensor2disp, tensor2rgb, vls_ins
from posenet import Posenet
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

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

class Logger:
    def __init__(self, logpath):
        self.logpath = logpath
        self.writer = None

    def create_summarywriter(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.logpath)

    def write_vls(self, data_blob, outputs, flowselector, reprojselector, step):
        img1 = data_blob['img1'][0].permute([1, 2, 0]).numpy().astype(np.uint8)
        img2 = data_blob['img2'][0].permute([1, 2, 0]).numpy().astype(np.uint8)
        insmap = data_blob['insmap'][0].squeeze().numpy()

        figmask_flow = tensor2disp(flowselector, vmax=1, viewind=0)
        figmask_reprojection = tensor2disp(reprojselector, vmax=1, viewind=0)
        insvls = vls_ins(img1, insmap)

        depthpredvls = tensor2disp(1 / outputs[('depth', 2)], vmax=0.15, viewind=0)
        depthgtvls = tensor2disp(1 / data_blob['depthmap'], vmax=0.15, viewind=0)
        flowvls = flow_to_image(outputs[('flowpred', 2)][0].detach().cpu().permute([1, 2, 0]).numpy(), rad_max=10)
        imgrecon = tensor2rgb(outputs[('reconImg', 2)], viewind=0)

        img_val_up = np.concatenate([np.array(insvls), np.array(img2)], axis=1)
        img_val_mid1 = np.concatenate([np.array(figmask_flow), np.array(figmask_reprojection)], axis=1)
        img_val_mid2 = np.concatenate([np.array(depthpredvls), np.array(depthgtvls)], axis=1)
        img_val_mid3 = np.concatenate([np.array(imgrecon), np.array(flowvls)], axis=1)
        img_val = np.concatenate([np.array(img_val_up), np.array(img_val_mid1), np.array(img_val_mid2), np.array(img_val_mid3)], axis=0)
        self.writer.add_image('predvls', (torch.from_numpy(img_val).float() / 255).permute([2, 0, 1]), step)

        X = self.vls_sampling(np.array(insvls), img2, data_blob['depthvls'], data_blob['flowmap'], data_blob['insmap'], outputs)
        self.writer.add_image('X', (torch.from_numpy(X).float() / 255).permute([2, 0, 1]), step)

    def vls_sampling(self, img1, img2, depthgt, flowmap, insmap, outputs):
        depthgtnp = depthgt[0].squeeze().cpu().numpy()
        insmapnp = insmap[0].squeeze().cpu().numpy()
        flowmapnp = flowmap[0].cpu().numpy()

        h, w, _ = img1.shape
        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        selector = (depthgtnp > 0)

        flowx = outputs[('flowpred', 2)][0, 0].detach().cpu().numpy()
        flowy = outputs[('flowpred', 2)][0, 1].detach().cpu().numpy()
        flowxf = flowx[selector]
        flowyf = flowy[selector]

        floworgx = outputs['org_flow'][0, 0].detach().cpu().numpy()
        floworgy = outputs['org_flow'][0, 1].detach().cpu().numpy()
        floworgxf = floworgx[selector]
        floworgyf = floworgy[selector]

        xxf = xx[selector]
        yyf = yy[selector]
        df = depthgtnp[selector]

        slRange_sel = (np.mod(xx, 4) == 0) * (np.mod(yy, 4) == 0) * selector * (insmapnp > 0)
        dsratio = 4
        if np.sum(slRange_sel) > 0:
            xxfsl = xx[slRange_sel]
            yyfsl = yy[slRange_sel]
            rndidx = np.random.randint(0, xxfsl.shape[0], 1).item()

            xxfsl_sel = xxfsl[rndidx]
            yyfsl_sel = yyfsl[rndidx]

            slvlsxx_fg = (outputs['sample_pts'][0, :, int(yyfsl_sel / dsratio), int(xxfsl_sel / dsratio), 0].detach().cpu().numpy() + 1) / 2 * w
            slvlsyy_fg = (outputs['sample_pts'][0, :, int(yyfsl_sel / dsratio), int(xxfsl_sel / dsratio), 1].detach().cpu().numpy() + 1) / 2 * h
        else:
            slvlsxx_fg = None
            slvlsyy_fg = None

        slRange_sel = (np.mod(xx, 4) == 0) * (np.mod(yy, 4) == 0) * selector * (insmapnp == 0)
        if np.sum(slRange_sel) > 0:
            xxfsl = xx[slRange_sel]
            yyfsl = yy[slRange_sel]
            rndidx = np.random.randint(0, xxfsl.shape[0], 1).item()

            xxfsl_sel = xxfsl[rndidx]
            yyfsl_sel = yyfsl[rndidx]

            slvlsxx_bg = (outputs['sample_pts'][0, :, int(yyfsl_sel / dsratio), int(xxfsl_sel / dsratio), 0].detach().cpu().numpy() + 1) / 2 * w
            slvlsyy_bg = (outputs['sample_pts'][0, :, int(yyfsl_sel / dsratio), int(xxfsl_sel / dsratio), 1].detach().cpu().numpy() + 1) / 2 * h

            gtposx = xxfsl_sel + flowmapnp[0, yyfsl_sel, xxfsl_sel]
            gtposy = yyfsl_sel + flowmapnp[0, yyfsl_sel, xxfsl_sel]
        else:
            slvlsxx_bg = None
            slvlsyy_bg = None


        cm = plt.get_cmap('magma')
        rndcolor = cm(1 / df / 0.15)[:, 0:3]

        fig = plt.figure(figsize=(16, 9))
        canvas = FigureCanvasAgg(fig)
        fig.add_subplot(2, 2, 1)
        plt.scatter(xxf, yyf, 3, rndcolor)
        plt.imshow(img1)
        plt.title("Input")

        fig.add_subplot(2, 2, 2)
        plt.scatter(xxf + floworgxf, yyf + floworgyf, 3, rndcolor)
        plt.imshow(img2)
        plt.title("Original Prediction")

        fig.add_subplot(2, 2, 3)
        plt.scatter(xxf + flowxf, yyf + flowyf, 3, rndcolor)
        plt.imshow(img2)
        plt.title("Fixed Prediction")

        fig.add_subplot(2, 2, 4)
        if slvlsxx_fg is not None and slvlsyy_fg is not None:
            plt.scatter(slvlsxx_fg, slvlsyy_fg, 3, 'b')
            plt.scatter(slvlsxx_fg[16], slvlsyy_fg[16], 3, 'g')
        if slvlsxx_fg is not None and slvlsyy_fg is not None:
            plt.scatter(slvlsxx_bg, slvlsyy_bg, 3, 'b')
            plt.scatter(slvlsxx_bg[16], slvlsyy_bg[16], 3, 'g')
            plt.scatter(gtposx, gtposy, 3, 'r')
        plt.imshow(img2)
        plt.title("Sampling Arae")

        fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
        canvas.draw()
        buf = canvas.buffer_rgba()
        plt.close()
        X = np.asarray(buf)
        # Image.fromarray(X).show()
        return X

    def write_vls_eval(self, data_blob, outputs, tagname, step):
        img1 = data_blob['img1'][0].permute([1, 2, 0]).numpy().astype(np.uint8)
        img2 = data_blob['img2'][0].permute([1, 2, 0]).numpy().astype(np.uint8)
        insmap = data_blob['insmap'][0].squeeze().numpy()

        insvls = vls_ins(img1, insmap)

        depthpredvls = tensor2disp(1 / outputs[('depth', 2)], vmax=0.15, viewind=0)
        depthgtvls = tensor2disp(1 / data_blob['depthmap'], vmax=0.15, viewind=0)
        flowvls = flow_to_image(outputs[('flowpred', 2)][0].detach().cpu().permute([1, 2, 0]).numpy(), rad_max=10)
        imgrecon = tensor2rgb(outputs[('reconImg', 2)], viewind=0)

        img_val_up = np.concatenate([np.array(insvls), np.array(img2)], axis=1)
        img_val_mid2 = np.concatenate([np.array(depthpredvls), np.array(depthgtvls)], axis=1)
        img_val_mid3 = np.concatenate([np.array(imgrecon), np.array(flowvls)], axis=1)
        img_val = np.concatenate([np.array(img_val_up), np.array(img_val_mid2), np.array(img_val_mid3)], axis=0)
        self.writer.add_image('{}_predvls'.format(tagname), (torch.from_numpy(img_val).float() / 255).permute([2, 0, 1]), step)

        X = self.vls_sampling(np.array(insvls), img2, data_blob['depthvls'], data_blob['flowmap'], data_blob['insmap'], outputs)
        self.writer.add_image('{}_X'.format(tagname), (torch.from_numpy(X).float() / 255).permute([2, 0, 1]), step)

    def write_dict(self, results, step):
        for key in results:
            self.writer.add_scalar(key, results[key], step)

    def close(self):
        self.writer.close()

@torch.no_grad()
def validate_kitti(model, args, eval_loader, logger, group, total_steps):
    """ Peform validation using the KITTI-2015 (train) split """
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    gpu = args.gpu
    eval_epe = torch.zeros(2).cuda(device=gpu)
    eval_out = torch.zeros(2).cuda(device=gpu)

    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        image1 = data_blob['img1'].cuda(gpu) / 255.0
        image2 = data_blob['img2'].cuda(gpu) / 255.0
        intrinsic = data_blob['intrinsic'].cuda(gpu)
        insmap = data_blob['insmap'].cuda(gpu)
        flowgt = data_blob['flowmap'].cuda(gpu)
        depthpred = data_blob['depthpred'].cuda(gpu)
        posepred = data_blob['posepred'].cuda(gpu)

        outputs = model(image1, image2, depthpred, intrinsic, posepred, insmap)

        flow_pr = outputs[('flowpred', 2)]
        selector = (((flowgt[:, 0] == 0) * (flowgt[:, 1] == 0)) == 0).float().unsqueeze(1)

        epe = torch.sum((flow_pr - flowgt)**2, dim=1, keepdim=True).sqrt()
        mag = (torch.sum(flowgt**2, dim=1, keepdim=True) + 1e-10).sqrt()

        out = ((epe > 3.0) * ((epe / (mag + 1e-10)) > 0.05) * selector).float()

        eval_out[0] += torch.sum(out)
        eval_out[1] += torch.sum(selector)

        eval_epe[0] += torch.sum(torch.sum(epe * selector, dim=[1, 2, 3]) / torch.sum(selector, dim=[1, 2, 3]))
        eval_epe[1] += image1.shape[0]

        if not(logger is None) and np.mod(val_id, 20) == 0:
            seq, frmidx = data_blob['tag'][0].split(' ')
            tag = "{}_{}".format(seq.split('/')[-1], frmidx)
            logger.write_vls_eval(data_blob, outputs, tag, total_steps)

    if args.distributed:
        dist.all_reduce(tensor=eval_out, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=eval_epe, op=dist.ReduceOp.SUM, group=group)

    if args.gpu == 0:
        eval_out[0] = eval_out[0] / eval_out[1]
        eval_epe[0] = eval_epe[0] / eval_epe[1]

        print("in {} eval samples: Out: {:7.3f}, Epe: {:7.3f}".format(eval_epe[1].item(), eval_out[0].item(), eval_epe[0].item()))
        return {'out': float(eval_out[0].item()), 'epe': float(eval_epe[0].item())}
    else:
        return None

def read_splits():
    split_root = os.path.join(project_rootdir, 'exp_pose_mdepth_kitti_eigen/splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'train_files.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'test_files.txt'), 'r')]
    return train_entries, evaluation_entries

def get_rdepth_loss(depthgt, insmap, outputs, model):
    _, _, h, w = depthgt.shape

    maxinsnum = insmap.max().item() + 1

    selector = (depthgt > 0).float().squeeze(1).unsqueeze(-1).unsqueeze(-1)
    insnum_gt = model.module.eppcompress(insmap, selector, maxinsnum)
    depthgt_sum = model.module.eppcompress(insmap, depthgt.squeeze(1).unsqueeze(-1).unsqueeze(-1), maxinsnum)

    mean_gt = depthgt_sum / (insnum_gt + 1e-10)
    loss_depth = 0
    for k in range(1, 3, 1):
        depthpred_sum = model.module.eppcompress(insmap, outputs[('depth', k)].squeeze(1).unsqueeze(-1).unsqueeze(-1) * selector, maxinsnum)
        mean_pred = depthpred_sum / (insnum_gt + 1e-10)
        scale = (model.module.eppinflate(insmap, mean_gt / (mean_pred + 1e-10))).squeeze(-1).squeeze(-1).squeeze(1)
        depthpred_scaled = scale.detach() * outputs[('depth', k)]

        selector_loss = (depthgt > 0).float()
        loss_depth += torch.sum(torch.abs(depthpred_scaled - depthgt) * selector_loss) / torch.sum(selector_loss)

    return loss_depth / 2

def get_flow_loss(flowgt, outputs):
    selector = (((flowgt[:, 0] == 0) * (flowgt[:, 1] == 0)) == 0).float().unsqueeze(1)
    flowloss = 0
    for k in range(1, 3, 1):
        flowloss += torch.sum(torch.sum(torch.abs(outputs[('flowpred', k)] - flowgt), dim=1, keepdim=True) * selector) / (torch.sum(selector) + 1)
    flowloss = flowloss / 2
    return flowloss, selector

def get_reprojection_loss(img1, outputs, ssim):
    reprojloss = 0
    selector = (outputs[('reconImg', 2)].sum(dim=1, keepdim=True) != 0).float()
    for k in range(1, 3, 1):
        ssimloss = ssim(outputs[('reconImg', k)], img1).mean(dim=1, keepdim=True)
        l1_loss = torch.abs(outputs[('reconImg', k)] - img1).mean(dim=1, keepdim=True)
        reprojectionloss = 0.85 * ssimloss + 0.15 * l1_loss
        reprojloss += (reprojectionloss * selector).sum() / (selector.sum() + 1)
    reprojloss = reprojloss / 4
    return reprojloss, selector

def train(gpu, ngpus_per_node, args):
    print("Using GPU %d for training" % gpu)
    args.gpu = gpu

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=ngpus_per_node, rank=args.gpu)

    model = EppFlowNet(args=args)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(module=model)
        model = model.to(f'cuda:{args.gpu}')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True, output_device=args.gpu)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    ssim = SSIM()

    logroot = os.path.join(args.logroot, args.name)
    print("Parameter Count: %d, saving location: %s" % (count_parameters(model), logroot))

    if args.restore_ckpt is not None:
        print("=> loading checkpoint '{}'".format(args.restore_ckpt))
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.restore_ckpt, map_location=loc)
        model.load_state_dict(checkpoint, strict=False)

    model.train()

    train_entries, evaluation_entries = read_splits()

    train_dataset = KITTI_eigen(root=args.dataset_root, inheight=args.inheight, inwidth=args.inwidth, entries=train_entries, maxinsnum=args.maxinsnum,
                                depth_root=args.depth_root, depthvls_root=args.depthvlsgt_root, prediction_root=args.prediction_root, ins_root=args.ins_root, istrain=True, muteaug=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=int(args.num_workers / ngpus_per_node), drop_last=True, sampler=train_sampler)

    eval_dataset = KITTI_eigen(root=args.dataset_root, inheight=args.evalheight, inwidth=args.evalwidth, entries=evaluation_entries, maxinsnum=args.maxinsnum,
                               depth_root=args.depth_root, depthvls_root=args.depthvlsgt_root, prediction_root=args.prediction_root, ins_root=args.ins_root, istrain=False)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset) if args.distributed else None
    eval_loader = data.DataLoader(eval_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=3, drop_last=True, sampler=eval_sampler)

    print("Training splits contain %d images while test splits contain %d images" % (train_dataset.__len__(), eval_dataset.__len__()))

    if args.distributed:
        group = dist.new_group([i for i in range(ngpus_per_node)])

    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0

    if args.gpu == 0:
        logger = Logger(logroot)
        logger_evaluation = Logger(os.path.join(args.logroot, 'evaluation_eigen_background', args.name))
        logger.create_summarywriter()
        logger_evaluation.create_summarywriter()

    VAL_FREQ = 5000
    epoch = 0
    maxout = 100

    st = time.time()
    should_keep_training = True
    while should_keep_training:
        train_sampler.set_epoch(epoch)
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()

            image1 = data_blob['img1'].cuda(gpu) / 255.0
            image2 = data_blob['img2'].cuda(gpu) / 255.0
            intrinsic = data_blob['intrinsic'].cuda(gpu)
            insmap = data_blob['insmap'].cuda(gpu)
            depthgt = data_blob['depthmap'].cuda(gpu)
            flowgt = data_blob['flowmap'].cuda(gpu)
            depthpred = data_blob['depthpred'].cuda(gpu)
            posepred = data_blob['posepred'].cuda(gpu)

            outputs = model(image1, image2, depthpred, intrinsic, posepred, insmap)
            depthloss = get_rdepth_loss(depthgt, insmap, outputs, model)
            flowloss, flowselector = get_flow_loss(flowgt, outputs)
            ssimloss, reprojselector = get_reprojection_loss(image1, outputs, ssim)

            metrics = dict()
            metrics['depthloss'] = depthloss.item()
            metrics['flowloss'] = flowloss.item()
            metrics['ssimloss'] = ssimloss.item()

            loss = depthloss + flowloss + ssimloss * 0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()
            scheduler.step()

            if args.gpu == 0:
                logger.write_dict(metrics, step=total_steps)
                if total_steps % SUM_FREQ == 0:
                    dr = time.time() - st
                    resths = (args.num_steps - total_steps) * dr / (total_steps + 1) / 60 / 60
                    print("Step: %d, rest hour: %f, depthloss: %f, flowloss: %f, ssimloss: %f" % (total_steps, resths, depthloss.item(), flowloss.item(), ssimloss.item()))
                    logger.write_vls(data_blob, outputs, flowselector, reprojselector, total_steps)

            if total_steps % VAL_FREQ == 1:
                if args.gpu == 0:
                    results = validate_kitti(model.module, args, eval_loader, logger, group, total_steps)
                else:
                    results = validate_kitti(model.module, args, eval_loader, None, group, None)

                if args.gpu == 0:
                    logger_evaluation.write_dict(results, total_steps)
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
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--inheight', type=int, default=320)
    parser.add_argument('--inwidth', type=int, default=960)
    parser.add_argument('--evalheight', type=int, default=320)
    parser.add_argument('--evalwidth', type=int, default=1216)
    parser.add_argument('--maxinsnum', type=int, default=50)
    parser.add_argument('--min_depth_pred', type=float, default=1)
    parser.add_argument('--max_depth_pred', type=float, default=85)
    parser.add_argument('--min_depth_eval', type=float, default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, default=80)

    parser.add_argument('--tscale_range', type=float, default=3)
    parser.add_argument('--objtscale_range', type=float, default=10)
    parser.add_argument('--angx_range', type=float, default=0.03)
    parser.add_argument('--angy_range', type=float, default=0.06)
    parser.add_argument('--angz_range', type=float, default=0.01)
    parser.add_argument('--num_layers', type=int, default=50)
    parser.add_argument('--num_deges', type=int, default=32)
    parser.add_argument('--maxlogscale', type=float, default=1.5)

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--semantics_root', type=str)
    parser.add_argument('--depth_root', type=str)
    parser.add_argument('--depthvlsgt_root', type=str)
    parser.add_argument('--prediction_root', type=str)
    parser.add_argument('--ins_root', type=str)
    parser.add_argument('--logroot', type=str)
    parser.add_argument('--num_workers', type=int, default=12)

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