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
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader
from exp_poses.dataset_kitti_eigen_poseselector import KITTI_eigen
from exp_poses.dataset_kitti_odom_poseselector_generation import KITTI_odom
from exp_poses.eppflownet.EppFlowNet_poseselector import EppFlowNet

from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from PIL import Image, ImageDraw
from core.utils.flow_viz import flow_to_image
from core.utils.utils import InputPadder, forward_interpolate, tensor2disp, tensor2rgb, vls_ins
from posenet import Posenet
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable
import glob

from tqdm import tqdm

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def readlines(filename):
    with open(filename, 'r') as f:
        filenames = f.readlines()
    return filenames

def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data

def get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam):
    pose_imu2cam = np.eye(4)
    pose_imu2cam[0:3, 0:3] = np.reshape(imu2cam['R'], [3, 3])
    pose_imu2cam[0:3, 3] = imu2cam['T']

    pose_velo2cam = np.eye(4)
    pose_velo2cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
    pose_velo2cam[0:3, 3] = velo2cam['T']

    R_rect_00 = np.eye(4)
    R_rect_00[0:3, 0:3] = cam2cam['R_rect_00'].reshape(3, 3)

    intrinsic = np.eye(4)
    intrinsic[0:3, 0:3] = cam2cam['P_rect_02'].reshape(3, 4)[0:3, 0:3]

    org_intrinsic = np.eye(4)
    org_intrinsic[0:3, :] = cam2cam['P_rect_02'].reshape(3, 4)
    extrinsic_from_intrinsic = np.linalg.inv(intrinsic) @ org_intrinsic
    extrinsic_from_intrinsic[0:3, 0:3] = np.eye(3)

    extrinsic = extrinsic_from_intrinsic @ R_rect_00 @ pose_velo2cam @ pose_imu2cam

    return intrinsic.astype(np.float32), extrinsic.astype(np.float32)

def read_deepv2d_pose(deepv2dpose_path):
    # Read Pose from Deepv2d
    posesstr = readlines(deepv2dpose_path)
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
    return pose_deepv2d

@torch.no_grad()
def validate_kitti(model, args, eval_loader):
    """ Peform validation using the KITTI-2015 (train) split """
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    gpu = args.gpu
    export_root = get_export_name(args)
    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        image1 = data_blob['img1'].cuda(gpu) / 255.0
        image2 = data_blob['img2'].cuda(gpu) / 255.0
        intrinsic = data_blob['intrinsic'].cuda(gpu)
        insmap = data_blob['insmap'].cuda(gpu)
        mD_pred = data_blob['mdDepth_pred'].cuda(gpu)
        ang_decps_pad = data_blob['ang_decps_pad'].cuda(gpu)
        scl_decps_pad = data_blob['scl_decps_pad'].cuda(gpu)
        mvd_decps_pad = data_blob['mvd_decps_pad'].cuda(gpu)
        posepred = data_blob['posepred'].cuda(gpu)
        tag = data_blob['tag'][0]

        mD_pred_clipped = torch.clamp_min(mD_pred, min=args.min_depth_pred)

        outputs = model(image1, image2, mD_pred_clipped, intrinsic, posepred, ang_decps_pad, scl_decps_pad, mvd_decps_pad, insmap)
        poseselected = outputs[('afft_all', 2)][0, -1, 0]

        seq = tag.split(' ')[0].split('/')[1][0:21]
        frmid = tag.split(' ')[1]

        RANSAC_pose_path = os.path.join(args.RANSACPose_root, "000", seq[0:10], seq + "_sync", 'image_02', "{}.pickle".format(str(frmid).zfill(10)))
        RANSAC_pose = pickle.load(open(RANSAC_pose_path, "rb"))

        poseselected_np = poseselected.cpu().numpy()
        pose_bs_np = RANSAC_pose @ np.linalg.inv(RANSAC_pose[0]) @ poseselected_np
        pose_bs_np[0] = poseselected_np

        export_folder = os.path.join(export_root, seq[0:10], seq + "_sync", 'image_02')
        os.makedirs(export_folder, exist_ok=True)
        export_path = os.path.join(export_folder,  "{}.pickle".format(str(frmid).zfill(10)))
        with open(export_path, 'wb') as handle:
            pickle.dump(pose_bs_np, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

def get_odomentries(args):
    import glob
    odomentries = list()
    odomseqs = [
        '2011_10_03/2011_10_03_drive_0027_sync',
        '2011_09_30/2011_09_30_drive_0016_sync',
        '2011_09_30/2011_09_30_drive_0018_sync',
        '2011_09_30/2011_09_30_drive_0027_sync'
    ]
    for odomseq in odomseqs:
        leftimgs = glob.glob(os.path.join(args.odomroot, odomseq, 'image_02/data', "*.png"))
        for leftimg in leftimgs:
            imgname = os.path.basename(leftimg)
            odomentries.append("{} {} {}".format(odomseq, imgname.rstrip('.png'), 'l'))
    return odomentries

def read_splits(args):
    split_root = os.path.join(project_rootdir, 'exp_pose_mdepth_kitti_eigen/splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'train_files.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'test_files.txt'), 'r')]
    odom_entries = get_odomentries(args)

    entries = train_entries
    folds = list()
    for entry in entries:
        seq, idx, _ = entry.split(' ')
        folds.append(seq)
    folds = list(set(folds))

    entries_expand = list()
    for fold in folds:
        pngs = glob.glob(os.path.join(args.dataset_root, fold, 'image_02/data/*.png'))
        for png in pngs:
            frmidx = png.split('/')[-1].split('.')[0]
            entry_expand = "{} {} {}".format(fold, frmidx.zfill(10), 'l')
            entries_expand.append(entry_expand)

    tot_entries = odom_entries + entries_expand + evaluation_entries

    export_root = get_export_name(args)
    ungenerated_entries = list()
    for entry in tot_entries:
        seq, frmidx, _ = entry.split(' ')
        if not os.path.exists(os.path.join(export_root, seq, "{}.pickle".format(str(frmidx).zfill(10)))):
            ungenerated_entries.append(entry)
    return ungenerated_entries


def get_reprojection_loss(img1, outputs, ssim, args):
    rpjloss = 0
    _, _, h, w = img1.shape
    selector = (outputs[('img1_recon', 2)][:, -1].sum(dim=1, keepdim=True) != 0).float()
    selector[:, :, 0:int(0.25810811 * h)] = 0
    for k in range(1, 3, 1):
        recimg = outputs[('img1_recon', k)].squeeze(1)
        # tensor2rgb(recimg, viewind=0).show()
        # tensor2disp(selector, viewind=0, vmax=1).show()
        ssimloss = ssim(recimg, img1).mean(dim=1, keepdim=True)
        l1_loss = torch.abs(recimg - img1).mean(dim=1, keepdim=True)
        rpjloss_c = 0.85 * ssimloss + 0.15 * l1_loss
        rpjloss_cm = (rpjloss_c * selector).sum() / (selector.sum() + 1)
        rpjloss += rpjloss_cm
    rpjloss = rpjloss / 2
    return rpjloss

def get_scale_loss(outputs, gpsscale):
    scaleloss = 0
    for k in range(1, 3, 1):
        scale_pred = outputs[('scale_adj', k)]
        scaleloss += torch.abs(gpsscale.unsqueeze(1) - scale_pred[:, :, 0, 0])
    scaleloss = scaleloss / 2
    return scaleloss

def get_posel1_loss(outputs, gpsscale):
    scaleloss = 0
    for k in range(1, 3, 1):
        scale_pred = outputs[('scale_adj', k)]
        scaleloss += torch.abs(gpsscale.unsqueeze(1) - scale_pred[:, :, 0, 0])
    scaleloss = scaleloss / 2
    return scaleloss

def get_seq_loss(IMUlocations1, leftarrs1, rightarrs1, IMUlocations2, leftarrs2, rightarrs2, outputs, args):
    seqloss_scale = 0
    seqloss_fin = 0
    for k in range(1, 3, 1):
        poses_pred = outputs[('afft_all', k)][:, :, 0, :, :]
        poses_pred_list = torch.split(poses_pred, dim=1, split_size_or_sections=1)
        for m in range(len(poses_pred_list)):
            pose_pred = poses_pred_list[m]
            pos_pred_forwaed = torch.inverse(leftarrs1 @ pose_pred @ rightarrs1)[:, :, 0:3, 3:4]
            seqloss_scale_forward = torch.mean(torch.abs(pos_pred_forwaed - IMUlocations1))

            pos_pred_backwaed = torch.inverse(leftarrs2 @ torch.inverse(pose_pred) @ rightarrs2)[:, :, 0:3, 3:4]
            seqloss_scale_backward = torch.mean(torch.abs(pos_pred_backwaed - IMUlocations2))

            seqloss_scale_c = (seqloss_scale_forward + seqloss_scale_backward) / 2
            if m == len(poses_pred_list) - 1:
                seqloss_fin += seqloss_scale_c
            else:
                seqloss_scale += seqloss_scale_c

    seqloss_scale = seqloss_scale / 2 / args.num_angs
    seqloss_fin = seqloss_fin / 2
    return (seqloss_scale + seqloss_fin) / 2

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

    all_ckpts = glob.glob(os.path.join(args.restore_ckpt, "*.pth"))
    all_ckpts.sort()

    print("=> loading checkpoint '{}'".format(args.restore_ckpt))
    loc = 'cuda:{}'.format(args.gpu)
    checkpoint = torch.load(args.restore_ckpt, map_location=loc)
    model.load_state_dict(checkpoint, strict=False)

    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'eppflownet/pose_bin8.pickle'), 'rb') as f:
        linlogdedge = pickle.load(f)

    model.train()

    entries = read_splits(args)

    interval = np.floor(len(entries) / ngpus_per_node).astype(np.int).item()
    if interval == 0:
        return

    if args.gpu == ngpus_per_node - 1:
        stidx = int(interval * args.gpu)
        edidx = len(entries)
    else:
        stidx = int(interval * args.gpu)
        edidx = int(interval * (args.gpu + 1))

    print("GPU %d, fromm %d to %d, in total %d" % (gpu, stidx, edidx, len(entries[stidx:edidx])))

    eval_dataset = KITTI_odom(root=args.dataset_root, odomroot=args.odomroot ,inheight=args.evalheight, inwidth=args.evalwidth, entries=entries[stidx:edidx], maxinsnum=args.maxinsnum, linlogdedge=linlogdedge, num_samples=args.num_angs,
                              prediction_root=args.prediction_root, ins_root=args.ins_root, mdPred_root=args.mdPred_root,
                              RANSACPose_root=args.RANSACPose_root, istrain=False, isgarg=True)

    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=True, num_workers=3, drop_last=False, shuffle=False)

    validate_kitti(model.module, args, eval_loader)

    return

def get_export_name(args):
    modelname = args.restore_ckpt.split('/')[-2]
    exportname = modelname + '_selected'
    export_folder = os.path.join(args.export_root, exportname)
    return export_folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint")

    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--inheight', type=int, default=320)
    parser.add_argument('--inwidth', type=int, default=960)
    parser.add_argument('--evalheight', type=int, default=320)
    parser.add_argument('--evalwidth', type=int, default=1216)
    parser.add_argument('--maxinsnum', type=int, default=50)
    parser.add_argument('--min_depth_pred', type=float, default=1)
    parser.add_argument('--maxlogscale', type=float, default=1)

    parser.add_argument('--num_scales', type=int, default=8)
    parser.add_argument('--num_angs', type=int, default=4)

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--semantics_root', type=str)
    parser.add_argument('--depth_root', type=str)
    parser.add_argument('--depthvlsgt_root', type=str)
    parser.add_argument('--prediction_root', type=str, default=None)
    parser.add_argument('--mdPred_root', type=str)
    parser.add_argument('--RANSACPose_root', type=str)
    parser.add_argument('--deepv2dPose_root', type=str)
    parser.add_argument('--ins_root', type=str)
    parser.add_argument('--export_root', type=str)
    parser.add_argument('--enable_regeneration', action="store_true")
    parser.add_argument('--odomroot', type=str)
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

    if args.enable_regeneration:
        import shutil
        export_folder = get_export_name(args)
        shutil.rmtree(export_folder)

    if args.distributed:
        args.world_size = ngpus_per_node
        mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        train(args.gpu, ngpus_per_node, args)