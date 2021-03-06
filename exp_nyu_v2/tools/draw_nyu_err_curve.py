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
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from matplotlib.backends.backend_agg import FigureCanvasAgg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import pickle

from torch.utils.data import DataLoader
from exp_nyu_v2.dataset_nyu_fixation import NYUV2
from exp_nyu_v2.eppflowenet.EppFlowNet_scale_initialD_nyu import EppFlowNet

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

# import torch.backends.cudnn as cudnn
# cudnn.benchmark = True


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

def fetch_optimizer(args, model, steps_per_epoch):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=20, steps_per_epoch=steps_per_epoch, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler

class Logger:
    def __init__(self, logpath):
        self.logpath = logpath
        self.writer = None

    def create_summarywriter(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.logpath)

    def write_vls(self, data_blob, outputs, flowselector, step):
        img1 = data_blob['img1'][0].permute([1, 2, 0]).numpy().astype(np.uint8)
        img2 = data_blob['img2'][0].permute([1, 2, 0]).numpy().astype(np.uint8)

        figmask_flow = tensor2disp(flowselector, vmax=1, viewind=0)

        depthpredvls = tensor2disp(1 / outputs[('depth', 2)], vmax=1, viewind=0)
        flowvls = flow_to_image(outputs[('flowpred', 2)][0].detach().cpu().permute([1, 2, 0]).numpy(), rad_max=10)
        imgrecon = tensor2rgb(outputs[('reconImg', 2)], viewind=0)

        img_val_up = np.concatenate([np.array(img1), np.array(img2)], axis=1)
        img_val_mid2 = np.concatenate([np.array(depthpredvls), np.array(figmask_flow)], axis=1)
        img_val_mid3 = np.concatenate([np.array(imgrecon), np.array(flowvls)], axis=1)
        img_val = np.concatenate([np.array(img_val_up), np.array(img_val_mid2), np.array(img_val_mid3)], axis=0)
        self.writer.add_image('predvls', (torch.from_numpy(img_val).float() / 255).permute([2, 0, 1]), step)

        X = self.vls_sampling(img1, img2, data_blob['depthmap'], data_blob['flowpred'], outputs)
        self.writer.add_image('X', (torch.from_numpy(X).float() / 255).permute([2, 0, 1]), step)

    def vls_sampling(self, img1, img2, depthgt, flowpred, outputs):
        depthgtnp = depthgt[0].squeeze().cpu().numpy()

        h, w, _ = img1.shape
        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        selector = (depthgtnp > 0)

        slRange_sel = (np.mod(xx, 4) == 0) * (np.mod(yy, 4) == 0) * selector
        dsratio = 4

        xxfsl = xx[slRange_sel]
        yyfsl = yy[slRange_sel]
        rndidx = np.random.randint(0, xxfsl.shape[0], 1).item()

        xxfsl_sel = xxfsl[rndidx]
        yyfsl_sel = yyfsl[rndidx]

        slvlsxx_fg = (outputs['sample_pts'][0, :, int(yyfsl_sel / dsratio), int(xxfsl_sel / dsratio), 0].detach().cpu().numpy() + 1) / 2 * w
        slvlsyy_fg = (outputs['sample_pts'][0, :, int(yyfsl_sel / dsratio), int(xxfsl_sel / dsratio), 1].detach().cpu().numpy() + 1) / 2 * h

        flow_predx = flowpred[0, 0, yyfsl_sel, xxfsl_sel].cpu().numpy()
        flow_predy = flowpred[0, 1, yyfsl_sel, xxfsl_sel].cpu().numpy()

        fig = plt.figure(figsize=(16, 9))
        canvas = FigureCanvasAgg(fig)
        fig.add_subplot(2, 1, 1)
        plt.imshow(img1)
        plt.scatter(xxfsl_sel, yyfsl_sel, 3, 'r')
        plt.title("Input")

        fig.add_subplot(2, 1, 2)
        plt.scatter(slvlsxx_fg, slvlsyy_fg, 3, 'b')
        plt.scatter(xxfsl_sel + flow_predx, yyfsl_sel + flow_predy, 3, 'r')
        plt.imshow(img2)
        plt.title("Sampling Arae")

        fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
        canvas.draw()
        buf = canvas.buffer_rgba()
        plt.close()
        X = np.asarray(buf)
        return X

    def write_vls_eval(self, data_blob, outputs, tagname, step):
        img1 = data_blob['img1'][0].permute([1, 2, 0]).numpy().astype(np.uint8)
        img2 = data_blob['img2'][0].permute([1, 2, 0]).numpy().astype(np.uint8)

        inputdepth = tensor2disp(1 / data_blob['depthmap'], vmax=1, viewind=0)
        depthpredvls = tensor2disp(1 / outputs[('depth', 2)], vmax=1, viewind=0)
        flowvls = flow_to_image(outputs[('flowpred', 2)][0].detach().cpu().permute([1, 2, 0]).numpy(), rad_max=10)
        imgrecon = tensor2rgb(outputs[('reconImg', 2)], viewind=0)

        img_val_up = np.concatenate([np.array(img1), np.array(img2)], axis=1)
        img_val_mid2 = np.concatenate([np.array(inputdepth), np.array(depthpredvls)], axis=1)
        img_val_mid3 = np.concatenate([np.array(imgrecon), np.array(flowvls)], axis=1)
        img_val = np.concatenate([np.array(img_val_up), np.array(img_val_mid2), np.array(img_val_mid3)], axis=0)
        self.writer.add_image('{}_predvls'.format(tagname), (torch.from_numpy(img_val).float() / 255).permute([2, 0, 1]), step)

        X = self.vls_sampling(img1, img2, data_blob['depthmap'], data_blob['flowpred'], outputs)
        self.writer.add_image('{}_X'.format(tagname), (torch.from_numpy(X).float() / 255).permute([2, 0, 1]), step)

    def write_dict(self, results, step):
        for key in results:
            self.writer.add_scalar(key, results[key], step)

    def close(self):
        self.writer.close()

def scale_invariant(gt, pr):
    """
    Computes the scale invariant loss based on differences of logs of depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)
    depth1:  one depth map
    depth2:  another depth map
    Returns:
        scale_invariant_distance
    """
    gt = gt.reshape(-1)
    pr = pr.reshape(-1)

    v = gt > 0.1
    gt = gt[v]
    pr = pr[v]

    log_diff = np.log(gt) - np.log(pr)
    num_pixels = np.float32(log_diff.size)

    # sqrt(Eq. 3)
    return np.sqrt(np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(num_pixels))

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

    sc_inv = scale_invariant(gt, pred)
    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3, sc_inv]

@torch.no_grad()
def validate_kitti(model, args, eval_loader, group, isorg=False, domask=False, isave=False, bridge_entries=None, deepv2d_entries=None):
    """ Peform validation using the KITTI-2015 (train) split """
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    gpu = args.gpu
    eval_measures_depth = torch.zeros(11).cuda(device=gpu)
    maskednum = 0

    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        image1 = data_blob['img1'].cuda(gpu) / 255.0
        image2 = data_blob['img2'].cuda(gpu) / 255.0
        intrinsic = data_blob['intrinsic'].cuda(gpu)
        posepred = data_blob['posepred'].cuda(gpu)
        depthgt = data_blob['depthmap'].cuda(gpu)
        entry = data_blob['tag'][0]

        mD_pred = data_blob['mdDepth_pred'].cuda(gpu)
        mD_pred_clipped = torch.clamp_min(mD_pred, min=args.min_depth_pred)

        if not isorg:
            if domask and torch.sqrt(torch.sum(posepred.squeeze()[0:3, 3] ** 2)) < 1e-2:
                predread = mD_pred
                maskednum += 1
            else:
                outputs = model(image1, image2, mD_pred_clipped, intrinsic, posepred)
                predread = outputs[('depth', 2)]
        else:
            predread = mD_pred

        if isave:
            predread_np = predread.squeeze().cpu().numpy()
            predread_np = predread_np * 1000
            predread_np = predread_np.astype(np.uint16)
            deepv2didx = find_idx_name(entry, bridge_entries, deepv2d_entries)
            Image.fromarray(predread_np).save(os.path.join(args.export_root, str(deepv2didx).zfill(5) + '.png'))

            poseprednp = posepred.squeeze().cpu().numpy()
            with open(os.path.join(args.export_root, str(deepv2didx).zfill(5) + '.pickle'), 'wb') as handle:
                pickle.dump(poseprednp, handle, protocol=pickle.HIGHEST_PROTOCOL)

        selector = ((depthgt > 0) * (predread > 0) * (mD_pred > 0)).float()
        depth_gt_flatten = depthgt[selector == 1].cpu().numpy()
        pred_depth_flatten = predread[selector == 1].cpu().numpy()

        if args.mean_scale:
            pred_depth_flatten = pred_depth_flatten / np.mean(pred_depth_flatten) * np.mean(depth_gt_flatten)

        eval_measures_depth_np = compute_errors(gt=depth_gt_flatten, pred=pred_depth_flatten)

        eval_measures_depth[:10] += torch.tensor(eval_measures_depth_np).cuda(device=gpu)
        eval_measures_depth[10] += 1

    if args.distributed:
        dist.all_reduce(tensor=eval_measures_depth, op=dist.ReduceOp.SUM, group=group)

    if args.gpu == 0:
        eval_measures_depth[0:10] = eval_measures_depth[0:10] / eval_measures_depth[10]
        eval_measures_depth = eval_measures_depth.cpu().numpy()
        if not domask:
            print('Computing Depth errors for %f eval samples' % (eval_measures_depth[10].item()))
        else:
            print('Computing Depth errors for %f eval samples, masked: %f' % (eval_measures_depth[10].item(), maskednum))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3', 'sc_inv'))
        for i in range(9):
            print('{:7.3f}, '.format(eval_measures_depth[i]), end='')
        print('{:7.3f}'.format(eval_measures_depth[9]))

        return {'silog': float(eval_measures_depth[0]),
                'abs_rel': float(eval_measures_depth[1]),
                'log10': float(eval_measures_depth[2]),
                'rms': float(eval_measures_depth[3]),
                'sq_rel': float(eval_measures_depth[4]),
                'log_rms': float(eval_measures_depth[5]),
                'd1': float(eval_measures_depth[6]),
                'd2': float(eval_measures_depth[7]),
                'd3': float(eval_measures_depth[8]),
                'sc_inv': float(eval_measures_depth[9])
                }
    else:
        return None

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

def read_splits():
    split_root = os.path.join(project_rootdir, 'exp_nyu_v2/splits')
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'nyudepthv2_organized_test_files.txt'), 'r')]
    bridge_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'nyudepthv2_organized_test_files_sorted.txt'), 'r')]
    deepv2d_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'deepv2d_eigen_test_files.txt'), 'r')]
    return evaluation_entries, bridge_entries, deepv2d_entries

def find_idx_name(entry, bridge_entries, deepv2d_entries):
    seq, idx = entry.split(' ')
    cpentry = os.path.join(seq, "rgb_{}.png".format("00001"))
    cpentry = '/' + cpentry
    for k, bentry in enumerate(bridge_entries):
        if cpentry == bentry.split(' ')[0]:
            break
    assert bridge_entries[k].split(' ')[0] == cpentry
    deepv2d_translated_entry = bridge_entries[k].split(' ')[1]

    for kk, dentry in enumerate(deepv2d_entries):
        if deepv2d_translated_entry == dentry:
            break
    assert deepv2d_translated_entry == deepv2d_entries[kk]
    return kk



def get_reprojection_loss(img1, insmap, outputs, ssim):
    reprojloss = 0
    selector = ((outputs[('reconImg', 2)].sum(dim=1, keepdim=True) != 0) * (insmap > 0)).float()
    for k in range(1, 3, 1):
        ssimloss = ssim(outputs[('reconImg', k)], img1).mean(dim=1, keepdim=True)
        l1_loss = torch.abs(outputs[('reconImg', k)] - img1).mean(dim=1, keepdim=True)
        reprojectionloss = 0.85 * ssimloss + 0.15 * l1_loss
        reprojloss += (reprojectionloss * selector).sum() / (selector.sum() + 1)
    reprojloss = reprojloss / 2
    return reprojloss, selector

def get_depth_loss(depthgt, mD_pred, outputs, silog_criterion):
    _, _, h, w = depthgt.shape
    selector = (depthgt > 0) * (mD_pred > 0)
    depthloss = 0
    for k in range(1, 3, 1):
        tmpselector = (selector * (outputs[('depth', k)] > 0)).float()
        depthloss += silog_criterion.forward(outputs[('depth', k)], depthgt, selector.to(torch.bool))

    depthloss = depthloss / 2
    return depthloss, tmpselector

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

    if args.restore_ckpt is not None:
        print("=> loading checkpoint '{}'".format(args.restore_ckpt))
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.restore_ckpt, map_location=loc)
        model.load_state_dict(checkpoint, strict=False)

    evaluation_entries, bridge_entries, deepv2d_entries = read_splits()

    eval_dataset = NYUV2(root=args.dataset_root, inheight=args.evalheight, inwidth=args.evalwidth, entries=evaluation_entries, mdPred_root=args.mdPred_root,
                               RANSACPose_root=args.RANSACPose_root, flowpred_root=args.flowpred_root, istrain=False)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset) if args.distributed else None
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=True, num_workers=3, drop_last=False, sampler=eval_sampler)

    if args.distributed:
        group = dist.new_group([i for i in range(ngpus_per_node)])

    if args.gpu == 0:
        results = validate_kitti(model.module, args, eval_loader, group, isorg=False, isave=False)
    else:
        results = validate_kitti(model.module, args, eval_loader, group, isorg=False, isave=False)

    if args.gpu == 0:
        validate_kitti(model.module, args, eval_loader, group, isorg=False, domask=True, isave=True, bridge_entries=bridge_entries, deepv2d_entries=deepv2d_entries)
    else:
        validate_kitti(model.module, args, eval_loader, group, isorg=False, domask=True, isave=True, bridge_entries=bridge_entries, deepv2d_entries=deepv2d_entries)

    if args.gpu == 0:
        results = validate_kitti(model.module, args, eval_loader, group, isorg=True, isave=False)
    else:
        validate_kitti(model.module, args, eval_loader, group, isorg=True, isave=False)


    return

def crop_img(img, left, top, crph, crpw):
    img = np.array(img)
    if len(img.shape) == 3:
        img_cropped = img[top:top+crph, left:left+crpw, :]
    else:
        img_cropped = img[top:top + crph, left:left + crpw]
    return Image.fromarray(img_cropped)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--inheight', type=int, default=448)
    parser.add_argument('--inwidth', type=int, default=448)
    parser.add_argument('--evalheight', type=int, default=448)
    parser.add_argument('--evalwidth', type=int, default=640)
    parser.add_argument('--min_depth_pred', type=float, default=1e-3)
    parser.add_argument('--max_depth_pred', type=float, default=10)
    parser.add_argument('--min_depth_eval', type=float, default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, default=80)
    parser.add_argument('--variance_focus', type=float,
                        help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error',
                        default=0.85)
    parser.add_argument('--maxlogscale', type=float, default=0.8)
    parser.add_argument('--num_deges', type=int, default=32)
    parser.add_argument('--ban_static_in_trianing', action='store_true')

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--mdPred_root', type=str)
    parser.add_argument('--RANSACPose_root', type=str)
    parser.add_argument('--flowpred_root', type=str)
    parser.add_argument('--logroot', type=str)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--mean_scale', action="store_true")
    parser.add_argument('--export_root', type=str)

    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dist_url', type=str, help='url used to set up distributed training', default='tcp://127.0.0.1:1235')
    parser.add_argument('--dist_backend', type=str, help='distributed backend', default='nccl')

    args = parser.parse_args()
    args.dist_url = args.dist_url.rstrip('1235') + str(np.random.randint(2000, 3000, 1).item())

    torch.manual_seed(1234)
    np.random.seed(1234)

    torch.cuda.empty_cache()
    ngpus_per_node = torch.cuda.device_count()

    evaluation_entries, bridge_entries, deepv2d_entries = read_splits()
    nyu_gt = np.load('/media/shengjie/disk1/data/nyutest/nyu_groundtruth.npy')
    nyu_org_root = '/media/shengjie/disk1/data/nyuv2_organized'
    nyu_deepv2d_root = '/media/shengjie/disk1/data/nyutest/nyu'
    pred_root = '/media/shengjie/disk1/Prediction/ablation_depth_predictions'
    export_root = '/media/shengjie/disk1/visualization/nyu_vls_cp'
    os.makedirs(export_root, exist_ok=True)

    allidx_rec = list()
    crph = 448
    crpw = 640

    top = int((480 - crph) / 2)
    left = int((640 - crpw) / 2)

    err_rec = list()
    err_rec_deepv2d = list()
    err_rec_deepv2d2 = list()
    err_rec_md = list()
    mv_rec = list()

    errval = list()
    errval2 = list()

    erridx = -4
    for _, entry in (enumerate(tqdm(evaluation_entries))):
        seq, frmidx = entry.split(' ')
        deepv2didx = find_idx_name(entry, bridge_entries, deepv2d_entries)
        img = crop_img(Image.open(os.path.join(nyu_org_root, seq, 'rgb_00000.png')), left, top, crph, crpw)
        img2 = crop_img(Image.open(os.path.join(nyu_org_root, seq, 'rgb_00001.png')), left, top, crph, crpw)

        depth_deepv2d_eight = os.path.join(pred_root, 'deepv2d/nyuv2/eight_view/{}.png'.format(str(deepv2didx).zfill(5)))
        depth_deepv2d_eight = crop_img(Image.open(depth_deepv2d_eight), left, top, crph, crpw)
        depth_deepv2d_eight = np.array(depth_deepv2d_eight).astype(np.float32) / 1000

        depth_deepv2d_two = os.path.join(pred_root, 'deepv2d/nyuv2/two_view/{}.png'.format(str(deepv2didx).zfill(5)))
        depth_deepv2d_two = crop_img(Image.open(depth_deepv2d_two), left, top, crph, crpw)
        depth_deepv2d_two = np.array(depth_deepv2d_two).astype(np.float32) / 1000

        ours = os.path.join(pred_root, 'ours/nyuv2/{}.png'.format(str(deepv2didx).zfill(5)))
        ours = Image.open(ours)
        ours = np.array(ours).astype(np.float32) / 1000

        ours_init = os.path.join(pred_root, 'ours/nyuv2_init/{}.png'.format(str(deepv2didx).zfill(5)))
        ours_init = Image.open(ours_init)
        ours_init = np.array(ours_init).astype(np.float32) / 1000

        pose_pred = os.path.join(pred_root, 'ours/nyuv2/{}.pickle'.format(str(deepv2didx).zfill(5)))
        pose_pred = pickle.load(open(pose_pred, "rb"))
        scale = np.sqrt(np.sum(pose_pred[0:3, 3] ** 2))

        depth_gt = os.path.join(nyu_org_root, seq, 'sync_depth_00000.png')
        depth_gt = crop_img(Image.open(depth_gt), left, top, crph, crpw)
        depth_gt = np.array(depth_gt).astype(np.float32) / 1000
        selector = depth_gt > 0

        depth_gt_flatten = depth_gt[selector == 1]
        pred_depth_flatten = ours[selector == 1]
        pred_depth_flatten = pred_depth_flatten / np.mean(pred_depth_flatten) * np.mean(depth_gt_flatten)
        deepv2d_depth_flatten = depth_deepv2d_eight[selector == 1]
        deepv2d_depth_flatten = deepv2d_depth_flatten / np.mean(deepv2d_depth_flatten) * np.mean(depth_gt_flatten)
        deepv2d_depth_flatten2 = depth_deepv2d_two[selector == 1]
        deepv2d_depth_flatten2 = deepv2d_depth_flatten2 / np.mean(deepv2d_depth_flatten2) * np.mean(depth_gt_flatten)
        ours_init_flatten = ours_init[selector == 1]
        ours_init_flatten = ours_init_flatten / np.mean(ours_init_flatten) * np.mean(depth_gt_flatten)

        eval_measures_depth_np = compute_errors(gt=depth_gt_flatten, pred=pred_depth_flatten)
        eval_measures_depth_deepv2d_np = compute_errors(gt=depth_gt_flatten, pred=deepv2d_depth_flatten)
        eval_measures_depth_deepv2d_np2 = compute_errors(gt=depth_gt_flatten, pred=deepv2d_depth_flatten2)
        eval_measures_depth_md_np = compute_errors(gt=depth_gt_flatten, pred=ours_init_flatten)

        err_rec.append(eval_measures_depth_np[erridx])
        mv_rec.append(scale)
        err_rec_deepv2d.append(eval_measures_depth_deepv2d_np[erridx])
        err_rec_md.append(eval_measures_depth_md_np[erridx])
        err_rec_deepv2d2.append(eval_measures_depth_deepv2d_np2[erridx])

        errval.append(eval_measures_depth_deepv2d_np)
        errval2.append(eval_measures_depth_np)

    errval = np.stack(errval, axis=0)
    errval = np.mean(errval, axis=0)

    errval2 = np.stack(errval2, axis=0)
    errval2 = np.mean(errval2, axis=0)

    err_rec = np.array(err_rec)
    mv_rec = np.array(mv_rec)
    err_rec_deepv2d = np.array(err_rec_deepv2d)
    err_rec_md = np.array(err_rec_md)
    err_rec_deepv2d2 = np.array(err_rec_deepv2d2)

    check_dist = np.linspace(0, 0.18, 200)
    dist = 0.5

    plot_mv = list()
    plot_err = list()
    plot_std = list()
    for d in check_dist:
        d_low = d * (1 - dist)
        d_hig = d * (1 + dist)

        selector = (mv_rec >= d_low) * (mv_rec <= d_hig)
        if np.sum(selector) < 5:
            continue
        else:
            err1 = np.mean(err_rec[selector])
            err2 = np.mean(err_rec_deepv2d[selector])
            err3 = np.mean(err_rec_md[selector])
            err4 = np.mean(err_rec_deepv2d2[selector])

            std1 = np.std(err_rec[selector])
            std2 = np.std(err_rec_deepv2d[selector])
            std3 = np.std(err_rec_md[selector])
            std4 = np.std(err_rec_deepv2d2[selector])
            plot_err.append(np.array([err1, err2, err3, err4]))
            plot_std.append(np.array([std1, std2, std3, std4]))
            plot_mv.append(d)

    plot_err = np.stack(plot_err, axis=0)
    plot_mv = np.array(plot_mv)
    plot_std = np.stack(plot_std, axis=0)

    thickness = 0.025

    fig, ax = plt.subplots()
    plt.plot(plot_mv, plot_err[:, 0])
    plt.plot(plot_mv, plot_err[:, 1])
    plt.plot(plot_mv, plot_err[:, 2])
    plt.plot(plot_mv, plot_err[:, 3])
    ax.fill_between(plot_mv, plot_err[:, 0] - plot_std[:, 0] * thickness, plot_err[:, 0] + plot_std[:, 0] * thickness, alpha=0.5)
    ax.fill_between(plot_mv, plot_err[:, 1] - plot_std[:, 1] * thickness, plot_err[:, 1] + plot_std[:, 1] * thickness, alpha=0.5)
    ax.fill_between(plot_mv, plot_err[:, 2] - plot_std[:, 2] * thickness, plot_err[:, 2] + plot_std[:, 2] * thickness, alpha=0.5)
    ax.fill_between(plot_mv, plot_err[:, 3] - plot_std[:, 3] * thickness, plot_err[:, 3] + plot_std[:, 3] * thickness, alpha=0.5)
    plt.xlabel('scale')
    plt.ylabel('a1')
    plt.legend(['ours', 'DeepV2D Eight View', 'Adabins', 'DeepV2D Two View'], bbox_to_anchor=(0.1, 0.3))
    plt.savefig('/home/shengjie/Desktop/2.png', bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    plt.show()
