import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import core.datasets as datasets
from utils import flow_viz
from utils import frame_utils

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate
from numpy.random import default_rng
import matplotlib.pyplot as plt


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


def bresenham(x0, y0, x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

def compute_intersectx(x1, x2, y1, y2, y):
    if np.abs(y2 - y1) < 1e-6:
        return None
    else:
        return int((y - y2) * (x2 - x1) / (y2 - y1) + x2)

def compute_intersecty(x1, x2, y1, y2, x):
    if np.abs(x2 - x1) < 1e-6:
        return None
    else:
        return int((y2 - y1) / (x2 - x1) * (x - x2) + y2)

def check_inboud(x, y, w, h):
    if x is None or y is None:
        return False
    elif x >= 0 and x <= w - 1 and y >=0 and y <= h - 1:
        return True
    else:
        return False

def compute_intersectborder(x1, x2, y1, y2, w, h):
    xl = 0
    yl = compute_intersecty(x1, x2, y1, y2, xl)

    xr = w - 1
    yr = compute_intersecty(x1, x2, y1, y2, xr)

    yu = 0
    xu = compute_intersectx(x1, x2, y1, y2, yu)

    yd = h - 1
    xd = compute_intersectx(x1, x2, y1, y2, yd)

    ptslist = list()
    if check_inboud(xl, yl, w, h):
        ptslist.append(np.array([xl, yl]))
    if check_inboud(xr, yr, w, h):
        ptslist.append(np.array([xr, yr]))
    if check_inboud(xu, yu, w, h):
        ptslist.append(np.array([xu, yu]))
    if check_inboud(xd, yd, w, h):
        ptslist.append(np.array([xd, yd]))
    return ptslist

def tnp2disp(tnp, vmax=10):
    tnp = tnp.astype(np.float)
    cm = plt.get_cmap('magma')
    tnp = tnp / vmax
    tnp = (cm(tnp) * 255).astype(np.uint8)
    return Image.fromarray(tnp[:, :, 0:3])

def rnd_sampe_foreground_background(xx, yy, insmask, samplenum, validmask):
    mask = (insmask == 1) * validmask
    xxfore = xx[mask]
    yyfore = yy[mask]

    rndind = np.random.choice(range(np.sum(mask) - 1), samplenum, replace=False)
    rndforex = xxfore[rndind]
    rndforey = yyfore[rndind]

    mask = (insmask == 0) * validmask
    xxback = xx[mask]
    yyback = yy[mask]

    rndind = np.random.choice(range(np.sum(mask) - 1), samplenum * 3, replace=False)
    rndbackx = xxback[rndind]
    rndbacky = yyback[rndind]

    rndx = np.concatenate([rndforex, rndbackx])
    rndy = np.concatenate([rndforey, rndbacky])

    return rndx, rndy

def epipole_vote(xx, yy, flownp, image1_vlsnp, image2_vlsnp, flow_gt_vls_np, valid_gt_vls_np):
    insmap = Image.open('/home/shengjie/Documents/Data/KittiInstance/kitti_semantics/training/instance/000000_10.png')
    insmap = np.array(insmap) % 256
    insind = 9

    # tnp2disp((insmap == 9), vmax=1).show()

    samplenum = 100
    _, h, w = flownp.shape
    vote_panel_fore = np.zeros([h, w])
    vote_panel_back = np.zeros([h, w])
    # rndind = np.random.choice(range(w * h - 1), samplenum, replace=False)
    # sample_y = (rndind / w).astype(np.int)
    # sample_x = (rndind - sample_y * w).astype(np.int)
    # rndx = xx[sample_y, sample_x]
    # rndy = yy[sample_y, sample_x]
    validmask = np.ones([h, w]) == 1
    rndx, rndy = rnd_sampe_foreground_background(xx, yy, insmap==insind, samplenum, validmask)

    # rndx[0] = 215
    # rndy[0] = 278

    for i in range(samplenum):
        x1 = rndx[i]
        x2 = rndx[i] + flownp[0, rndy[i], rndx[i]]

        y1 = rndy[i]
        y2 = rndy[i] + flownp[1, rndy[i], rndx[i]]

        pts_val = compute_intersectborder(x1, x2, y1, y2, w, h)
        if len(pts_val) != 2:
            continue

        pts_vote = bresenham(pts_val[0][0], pts_val[0][1], pts_val[1][0], pts_val[1][1])

        for pts in pts_vote:
            vote_panel_fore[pts[1], pts[0]] += 1

        # plt.figure()
        # plt.imshow(image1_vlsnp)
        # plt.scatter(x1, y1, 1, 'r')
        # plt.scatter(x2, y2, 1, 'r')
        # for pts in pts_val:
        #     plt.scatter(pts[0], pts[1], 10, 'g')
        # for pts in pts_vote:
        #     plt.scatter(pts[0], pts[1], 1, 'b')
        #
        # plt.figure()
        # plt.imshow(image2_vlsnp)
        # plt.scatter(x2, y2, 1, 'r')

    fig1 = plt.figure()
    plt.imshow(image1_vlsnp)
    plt.scatter(rndx[0:samplenum], rndy[0:samplenum], 1, 'r')
    plt.scatter(rndx[samplenum::], rndy[samplenum::], 1, 'g')
    fig1.savefig('/home/shengjie/Desktop/2021_01/2021_01_08/sample.png', bbox_inches='tight', pad_inches=0)

    tnp2disp(vote_panel_fore, vmax=50).save("/home/shengjie/Desktop/2021_01/2021_01_08/foreground_est.png")

    for i in range(samplenum, rndx.shape[0]):
        x1 = rndx[i]
        x2 = rndx[i] + flownp[0, rndy[i], rndx[i]]

        y1 = rndy[i]
        y2 = rndy[i] + flownp[1, rndy[i], rndx[i]]

        pts_val = compute_intersectborder(x1, x2, y1, y2, w, h)
        if len(pts_val) != 2:
            continue

        pts_vote = bresenham(pts_val[0][0], pts_val[0][1], pts_val[1][0], pts_val[1][1])

        for pts in pts_vote:
            vote_panel_back[pts[1], pts[0]] += 1

    tnp2disp(vote_panel_back, vmax=50).save("/home/shengjie/Desktop/2021_01/2021_01_08/background_est.png")

    ## ============================= ##
    vote_panel_fore = np.zeros([h, w])
    vote_panel_back = np.zeros([h, w])
    rndx, rndy = rnd_sampe_foreground_background(xx, yy, insmap == insind, samplenum, valid_gt_vls_np == 1)

    for i in range(samplenum):
        x1 = rndx[i]
        x2 = rndx[i] + flow_gt_vls_np[0, rndy[i], rndx[i]]

        y1 = rndy[i]
        y2 = rndy[i] + flow_gt_vls_np[1, rndy[i], rndx[i]]

        pts_val = compute_intersectborder(x1, x2, y1, y2, w, h)
        if len(pts_val) != 2:
            continue

        pts_vote = bresenham(pts_val[0][0], pts_val[0][1], pts_val[1][0], pts_val[1][1])

        for pts in pts_vote:
            vote_panel_fore[pts[1], pts[0]] += 1

    tnp2disp(vote_panel_fore, vmax=50).save("/home/shengjie/Desktop/2021_01/2021_01_08/foreground_gt.png")

    for i in range(samplenum, rndx.shape[0]):
        x1 = rndx[i]
        x2 = rndx[i] + flow_gt_vls_np[0, rndy[i], rndx[i]]

        y1 = rndy[i]
        y2 = rndy[i] + flow_gt_vls_np[1, rndy[i], rndx[i]]

        pts_val = compute_intersectborder(x1, x2, y1, y2, w, h)
        if len(pts_val) != 2:
            continue

        pts_vote = bresenham(pts_val[0][0], pts_val[0][1], pts_val[1][0], pts_val[1][1])

        for pts in pts_vote:
            vote_panel_back[pts[1], pts[0]] += 1

    tnp2disp(vote_panel_back, vmax=50).save("/home/shengjie/Desktop/2021_01/2021_01_08/background_gt.png")
    a = 1

@torch.no_grad()
def validate_kitti_customized(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training', root='/home/shengjie/Documents/Data/Kitti/kitti_stereo/stereo15')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        flowT = flow.cpu()
        flownp = flowT.numpy()

        image1_vls = padder.unpad(image1[0]).cpu()
        image2_vls = padder.unpad(image2[0]).cpu()

        image1_vlsnp = image1_vls.permute([1, 2, 0]).cpu().numpy().astype(np.uint8)
        image2_vlsnp = image2_vls.permute([1, 2, 0]).cpu().numpy().astype(np.uint8)
        flow_gt_vls_np = flow_gt.cpu().numpy()
        valid_gt_vls_np = valid_gt.cpu().numpy()

        _, h, w = flowT.shape
        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        resampledxx = xx + flowT[0].cpu().numpy()
        resampledyy = yy + flowT[1].cpu().numpy()

        epipole_vote(xx, yy, flownp, image1_vlsnp, image2_vlsnp, flow_gt_vls_np, valid_gt_vls_np)

        resampledxx = ((resampledxx / (w - 1)) - 0.5) * 2
        resampledyy = ((resampledyy / (h - 1)) - 0.5) * 2
        resamplegrid = torch.stack([torch.from_numpy(resampledxx), torch.from_numpy(resampledyy)], dim=2).unsqueeze(0).float()
        image1_recon_vls = torch.nn.functional.grid_sample(input=image2_vls.unsqueeze(0), grid=resamplegrid, mode='bilinear', padding_mode='reflection')

        # rndx = np.random.randint(0, w)
        # rndy = np.random.randint(0, h)
        rndx = 215
        rndy = 278
        tarx = rndx + flownp[0, int(rndy), int(rndx)]
        tary = rndy + flownp[1, int(rndy), int(rndx)]

        plt.figure()
        plt.imshow(image1.squeeze().permute([1,2,0]).cpu().numpy().astype(np.uint8))
        plt.scatter(rndx, rndy, 1, 'r')

        plt.figure()
        plt.imshow(image2.squeeze().permute([1,2,0]).cpu().numpy().astype(np.uint8))
        plt.scatter(tarx, tary, 1, 'r')

        plt.figure()
        plt.imshow(image1_recon_vls.squeeze().permute([1,2,0]).cpu().numpy().astype(np.uint8))

        import PIL.Image as Image
        from core.utils.flow_viz import flow_to_image
        flowimg = flow_to_image(flow.permute([1,2,0]).cpu().numpy())
        Image.fromarray(flowimg).show()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

@torch.no_grad()
def validate_kitti(model, args, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training', root=args.dataset)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        validate_kitti(model.module, args)


