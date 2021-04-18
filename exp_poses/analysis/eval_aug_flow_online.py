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
from matplotlib.backends.backend_agg import FigureCanvasAgg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader
from exp_poses.dataset_kitti_stereo15_orged import KITTI_eigen_stereo15
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

from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

MAX_FLOW = 400

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

def get_gt_flow(depth, valid, intrinsic, rel_pose):
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

def color_transform(img1, img2, photo_aug):
    device = img1.device
    img1 = img1[0]
    img2 = img2[0]
    img1 = img1.permute([1, 2, 0]).cpu().numpy().astype(np.uint8)
    img2 = img2.permute([1, 2, 0]).cpu().numpy().astype(np.uint8)
    image_stack = np.concatenate([img1, img2], axis=0)
    image_stack = np.array(photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
    img1, img2 = np.split(image_stack, 2, axis=0)
    img1 = torch.from_numpy(img1).permute([2, 0, 1]).float().unsqueeze(0).cuda(device)
    img2 = torch.from_numpy(img2).permute([2, 0, 1]).float().unsqueeze(0).cuda(device)
    return img1, img2

def get_flowpred_variance(model, img1, img2, photo_aug_inf, photo_aug_var):
    seedcad = [2234, 3234, 4234, 5234]
    flow_prs = list()
    for s in seedcad:
        torch.manual_seed(s)
        np.random.seed(s)

        img1_aug, img2_aug = color_transform(img1, img2, photo_aug_var)
        flow_low, flow_pr = model(img1_aug, img2_aug, iters=24, test_mode=True)
        flow_prs.append(flow_pr)

    flow_prs = torch.cat(flow_prs, dim=0)
    flow_prs_var = torch.sqrt(torch.mean(torch.mean((flow_prs - torch.mean(flow_prs, dim=0, keepdim=True)) ** 2, dim=1, keepdim=True), dim=0, keepdim=True))

    torch.manual_seed(1234)
    np.random.seed(1234)
    img1_aug, img2_aug = color_transform(img1, img2, photo_aug_inf)
    _, flow_pr_inf = model(img1_aug, img2_aug, iters=24, test_mode=True)
    flow_mag = torch.sqrt(torch.sum(flow_pr_inf ** 2, dim=1, keepdim=True))

    flow_prs_var_normed = flow_prs_var / flow_mag

    # tensor2rgb(img1 / 255.0, viewind=0).show()
    # tensor2rgb(img1_aug / 255.0, viewind=0).show()
    # tensor2disp(flow_prs_var, vmax=1, viewind=0).show()
    # tensor2disp(flow_prs_var_normed, vmax=0.1, viewind=0).show()
    return flow_pr_inf, flow_prs_var_normed

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

def inf_pose_flow(img1, img2, flow_pr_inf, flow_prs_var_normed, insmap, depthmap, intrinsic, pid):
    insmap_np = insmap[0, 0].cpu().numpy()
    intrinsicnp = intrinsic[0].cpu().numpy()
    dummyh = 370
    samplenum = 50000
    _, _, h, w = img1.shape
    border_sel = np.zeros([h, w])
    border_sel[int(0.25810811 * dummyh) : int(0.99189189 * dummyh)] = 1
    varbar = 5e-3
    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')

    flow_pr_inf_x = flow_pr_inf[0, 0].cpu().numpy()
    flow_pr_inf_y = flow_pr_inf[0, 1].cpu().numpy()

    xx_nf = xx + flow_pr_inf_x
    yy_nf = yy + flow_pr_inf_y

    flow_prs_var_normed_np = flow_prs_var_normed.squeeze().cpu().numpy()
    depthmap_np = depthmap.squeeze().cpu().numpy()

    selector = (xx_nf > 0) * (xx_nf < w) * (yy_nf > 0) * (yy_nf < h) * (insmap_np == 0) * border_sel * (flow_prs_var_normed_np < varbar) * (depthmap_np > 0)
    selector = selector == 1

    if samplenum > np.sum(selector):
        samplenum = np.sum(selector)

    np.random.seed(int(time.time() + pid))
    rndidx = np.random.randint(0, np.sum(selector), samplenum)
    # print(rndidx[0])
    xx_idx_sel = xx[selector][rndidx]
    yy_idx_sel = yy[selector][rndidx]

    selvls = np.zeros([h, w])
    selvls[yy_idx_sel, xx_idx_sel] = 1

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
    coorespondedDepth = depthmap_np[selector][rndidx][inliers_mask]
    scale = depth2scale(pts1_inliers, pts2_inliers, intrinsicnp, R, t, coorespondedDepth)

    # Image.fromarray(flow_to_image(flow_pr_inf[0].cpu().permute([1, 2, 0]).numpy())).show()
    # tensor2disp(torch.from_numpy(selector).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()
    # tensor2disp(depthmap > 0, vmax=1, viewind=0).show()
    # tensor2disp(torch.from_numpy(selvls).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()

    return R, t, scale

@torch.no_grad()
def validate_RAFT_flow_pose(args, model, eval_loader, group, usestreodepth=False, scale_info_src=None):
    """ Peform validation using the KITTI-2015 (train) split """
    """ Peform validation using the KITTI-2015 (train) split """
    gpu = args.gpu
    var_num = 5
    eval_epe = torch.zeros(var_num + 1).cuda(device=gpu)
    eval_out = torch.zeros(var_num + 1).cuda(device=gpu)
    opt_out = torch.zeros(2).cuda(device=gpu)
    model.eval()
    jitterparam = 0.86
    photo_aug_inf = ColorJitter(brightness=jitterparam, contrast=jitterparam, saturation=jitterparam, hue=jitterparam / 3.14)

    jitterparam = 0.2
    photo_aug_var = ColorJitter(brightness=jitterparam, contrast=jitterparam, saturation=jitterparam, hue=jitterparam / 3.14)

    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        image1 = data_blob['img1'].cuda(gpu)
        image2 = data_blob['img2'].cuda(gpu)
        intrinsic = data_blob['intrinsic'].cuda(gpu)
        insmap = data_blob['insmap'].cuda(gpu)
        depthgt = data_blob['depthmap'].cuda(gpu)
        posegt = data_blob['rel_pose'].cuda(gpu)
        flowgt_stereo = data_blob['flowgt_stereo'].cuda(gpu)
        depthpred_deepv2d = data_blob['depthpred_deepv2d'].cuda(gpu)
        posepred_deepv2d = data_blob['posepred_deepv2d'].cuda(gpu)

        mdDepth_pred = data_blob['mdDepth_pred'].cuda(gpu)

        if not usestreodepth:
            valid_deepv2d = depthpred_deepv2d > 0
        else:
            valid_deepv2d = depthgt > 0

        if torch.sum(posegt) == 4:
            continue

        mag = torch.sum(flowgt_stereo**2, dim=1).sqrt()
        mag = mag.view(-1)
        val = (mag.view(-1) >= 0.5) * (mag.view(-1) < MAX_FLOW) * (insmap.squeeze(1) == 0).view(-1) * (valid_deepv2d).view(-1)
        if torch.sum(val) < 5e3:
            continue

        flow_pr_inf, flow_prs_var_normed = get_flowpred_variance(model, image1, image2, photo_aug_inf, photo_aug_var)

        best_out = 1e10
        for p in range(var_num):
            if scale_info_src == 'stereo_depth':
                R, t, scale = inf_pose_flow(image1, image2, flow_pr_inf, flow_prs_var_normed, insmap, depthgt, intrinsic, p + var_num * args.gpu)
            elif scale_info_src == 'deepv2d_depth':
                R, t, scale = inf_pose_flow(image1, image2, flow_pr_inf, flow_prs_var_normed, insmap, depthpred_deepv2d, intrinsic, p + var_num * args.gpu)
            elif scale_info_src == 'mdPred':
                R, t, scale = inf_pose_flow(image1, image2, flow_pr_inf, flow_prs_var_normed, insmap, mdDepth_pred, intrinsic, p + var_num * args.gpu)
            elif scale_info_src == 'deppv2d_pose':
                R, t, _ = inf_pose_flow(image1, image2, flow_pr_inf, flow_prs_var_normed, insmap, depthpred_deepv2d, intrinsic, p + var_num * args.gpu)
                scale = np.sqrt(np.sum(posepred_deepv2d[0, 0, 0:3, 3].cpu().numpy() ** 2))

            pose_RANSAC = torch.clone(posepred_deepv2d)
            pose_RANSAC[0, 0, 0:3, 0:3] = torch.from_numpy(R).float().cuda(intrinsic.device)
            pose_RANSAC[0, 0, 0:3, 3:4] = torch.from_numpy(t * scale).float().cuda(intrinsic.device)

            if not usestreodepth:
                valid_deepv2d = depthpred_deepv2d > 0
                flow_RANSAC = get_gt_flow(depthpred_deepv2d, valid_deepv2d, intrinsic, pose_RANSAC)
            else:
                valid_deepv2d = depthgt > 0
                flow_RANSAC = get_gt_flow(depthgt, valid_deepv2d, intrinsic, pose_RANSAC)

            flow_eval = flow_RANSAC
            epe = torch.sum((flowgt_stereo - flow_eval)**2, dim=1).sqrt()
            mag = torch.sum(flowgt_stereo**2, dim=1).sqrt()

            epe = epe.view(-1)
            mag = mag.view(-1)
            val = (mag.view(-1) >= 0.5) * (mag.view(-1) < MAX_FLOW) * (insmap.squeeze(1) == 0).view(-1) * (valid_deepv2d).view(-1)

            out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()

            if out[val].mean() < best_out:
                best_out = out[val].mean()

            eval_epe[p] += epe[val].mean()
            eval_out[p] += out[val].mean()

        opt_out[0] += best_out
        opt_out[1] += 1
        eval_epe[-1] += 1
        eval_out[-1] += 1

    if args.distributed:
        dist.all_reduce(tensor=eval_epe, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=eval_out, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=opt_out, op=dist.ReduceOp.SUM, group=group)

    if args.gpu == 0:
        eval_epe[0:var_num] = eval_epe[0:var_num] / eval_epe[-1]
        eval_epe = eval_epe.cpu().numpy()
        eval_out[0:var_num] = eval_out[0:var_num] / eval_out[-1]
        eval_out = eval_out.cpu().numpy()
        opt_out[0] = opt_out[0] / opt_out[1]
        opt_out = opt_out.cpu().numpy()

        ave_out = np.mean(eval_out[0:var_num])
        var_out = np.sqrt(np.sum((eval_out[0:var_num] - ave_out)**2))
        # optimal_out = np.mean(np.eval_out[0:var_num])

        if usestreodepth:
            depthsrcname = 'stereo_depth'
        else:
            depthsrcname = 'deepv2d_depth'
        print('Evaluate Init Pose on %f samples, depthmap: %s, scaleinfo: %s, ave_out: %f, var_out: %f, optimal_out: %f' % (eval_out[1].item(), depthsrcname, scale_info_src, ave_out, var_out, opt_out[0].item()))
        return None
    else:
        return None

@torch.no_grad()
def validate_RAFT_flow(args, model, eval_loader, group, usestreodepth=False):
    """ Peform validation using the KITTI-2015 (train) split """
    """ Peform validation using the KITTI-2015 (train) split """
    gpu = args.gpu
    eval_epe = torch.zeros(2).cuda(device=gpu)
    eval_out = torch.zeros(2).cuda(device=gpu)
    model.eval()
    jitterparam = 0.86
    photo_aug_inf = ColorJitter(brightness=jitterparam, contrast=jitterparam, saturation=jitterparam, hue=jitterparam / 3.14)

    jitterparam = 0.2
    photo_aug_var = ColorJitter(brightness=jitterparam, contrast=jitterparam, saturation=jitterparam, hue=jitterparam / 3.14)

    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        image1 = data_blob['img1'].cuda(gpu)
        image2 = data_blob['img2'].cuda(gpu)
        intrinsic = data_blob['intrinsic'].cuda(gpu)
        insmap = data_blob['insmap'].cuda(gpu)
        depthgt = data_blob['depthmap'].cuda(gpu)
        flowmap = data_blob['flowmap'].cuda(gpu)
        posegt = data_blob['rel_pose'].cuda(gpu)
        flowgt_stereo = data_blob['flowgt_stereo'].cuda(gpu)
        depthpred_deepv2d = data_blob['depthpred_deepv2d'].cuda(gpu)
        posepred_deepv2d = data_blob['posepred_deepv2d'].cuda(gpu)

        if not usestreodepth:
            valid_deepv2d = depthpred_deepv2d > 0
        else:
            valid_deepv2d = depthgt > 0

        if torch.sum(posegt) == 4:
            continue

        mag = torch.sum(flowgt_stereo**2, dim=1).sqrt()
        mag = mag.view(-1)
        val = (mag.view(-1) >= 0.5) * (mag.view(-1) < MAX_FLOW) * (insmap.squeeze(1) == 0).view(-1) * (valid_deepv2d).view(-1)
        if torch.sum(val) < 5e3:
            continue

        img1_aug, img2_aug = color_transform(image1, image2, photo_aug_inf)
        _, flow_pr_inf = model(img1_aug, img2_aug, iters=24, test_mode=True)

        flow_eval = flow_pr_inf
        epe = torch.sum((flowgt_stereo - flow_eval)**2, dim=1).sqrt()
        mag = torch.sum(flowgt_stereo**2, dim=1).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = (mag.view(-1) >= 0.5) * (mag.view(-1) < MAX_FLOW) * (insmap.squeeze(1) == 0).view(-1) * (valid_deepv2d).view(-1)

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()

        eval_epe[0] += epe[val].mean()
        eval_epe[1] += 1
        eval_out[0] += out[val].mean()
        eval_out[1] += 1

    if args.distributed:
        dist.all_reduce(tensor=eval_epe, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=eval_out, op=dist.ReduceOp.SUM, group=group)

    if args.gpu == 0:
        eval_epe[0] = eval_epe[0] / eval_epe[1]
        eval_epe = eval_epe.cpu().numpy()
        eval_out[0] = eval_out[0] / eval_out[1]
        eval_out = eval_out.cpu().numpy()
        print('Evaluate Deepv2d Pose on %f samples, out: %f' % (eval_out[1].item(), eval_out[0].item()))
        return None
    else:
        return None

def read_splits_mapping():
    evaluation_entries = []
    for m in range(200):
        seqname = "kittistereo15_{}/kittistereo15_{}_sync".format(str(m).zfill(6), str(m).zfill(6))
        evaluation_entries.append("{} {} {}".format(seqname, "10".zfill(10), 'l'))
    return evaluation_entries

def train(gpu, ngpus_per_node, args):
    print("Using GPU %d for training" % gpu)
    args.gpu = gpu

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=ngpus_per_node, rank=args.gpu)

    model = RAFT(args=args)
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

    evaluation_entries = read_splits_mapping()

    eval_dataset = KITTI_eigen_stereo15(root=args.dataset_stereo15_orgned_root, inheight=args.evalheight, inwidth=args.evalwidth, entries=evaluation_entries, mdPred_root=args.mdPred_root,
                                        maxinsnum=args.maxinsnum, istrain=True, isgarg=True, deepv2dpred_root=args.deepv2dpred_root, prediction_root=args.prediction_root,
                                        flowPred_root=args.flowPred_root)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset) if args.distributed else None
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=True, num_workers=3, drop_last=True, sampler=eval_sampler)

    print("Test splits contain %d images" % (eval_dataset.__len__()))

    if args.distributed:
        group = dist.new_group([i for i in range(ngpus_per_node)])

    # validate_RAFT_flow(args, model, eval_loader, group, usestreodepth=False)
    validate_RAFT_flow_pose(args, model, eval_loader, group, usestreodepth=False, scale_info_src='mdPred')
    # validate_RAFT_flow_pose(args, model, eval_loader, group, usestreodepth=True, scale_info_src='mdPred')
    # validate_RAFT_flow_pose(args, model, eval_loader, group, usestreodepth=False, scale_info_src='deepv2d_depth')
    # validate_RAFT_flow_pose(args, model, eval_loader, group, usestreodepth=True, scale_info_src='stereo_depth')
    # validate_RAFT_flow_pose(args, model, eval_loader, group, usestreodepth=False, scale_info_src='stereo_depth')
    # validate_RAFT_flow_pose(args, model, eval_loader, group, usestreodepth=True, scale_info_src='deppv2d_pose')
    # validate_RAFT_flow_pose(args, model, eval_loader, group, usestreodepth=False, scale_info_src='deppv2d_pose')
    # validate_RAFT_flow_pose(args, model, eval_loader, group, usestreodepth=False)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--dropout', type=float, default=0.0)

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

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--dataset_stereo15_orgned_root', type=str)
    parser.add_argument('--semantics_root', type=str)
    parser.add_argument('--depth_root', type=str)
    parser.add_argument('--depthvlsgt_root', type=str)
    parser.add_argument('--prediction_root', type=str)
    parser.add_argument('--deepv2dpred_root', type=str)
    parser.add_argument('--mdPred_root', type=str)
    parser.add_argument('--flowPred_root', type=str)
    parser.add_argument('--ins_root', type=str)
    parser.add_argument('--logroot', type=str)
    parser.add_argument('--num_workers', type=int, default=12)

    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dist_url', type=str, help='url used to set up distributed training', default='tcp://127.0.0.1:1235')
    parser.add_argument('--dist_backend', type=str, help='distributed backend', default='nccl')

    args = parser.parse_args()
    args.dist_url = args.dist_url.rstrip('1235') + str(np.random.randint(2000, 3000, 1).item())

    torch.manual_seed(1234)
    np.random.seed(1234)

    torch.cuda.empty_cache()

    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        args.world_size = ngpus_per_node
        mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        train(args.gpu, ngpus_per_node, args)