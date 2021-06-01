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
from exp_poses.ablation_tool.dataset_kitti_odom_poseselector_generation_aba import KITTI_odom
from exp_poses.eppflownet.EppflowNet_poseselector_selfonly_scaleonly import EppFlowNet

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

        seq = tag.split(' ')[0].split('/')[1][0:21]
        frmid = tag.split(' ')[1]

        export_folder = os.path.join(args.export_root, seq[0:10], seq + "_sync", 'image_02')
        os.makedirs(export_folder, exist_ok=True)
        export_path = os.path.join(export_folder,  "{}.pickle".format(str(frmid).zfill(10)))

        if os.path.exists(export_path):
            continue


        posepred = posepred[:, :, 0]
        ang_decps_pad = ang_decps_pad[:, :, 0]
        scl_decps_pad = scl_decps_pad[:, :, 0]
        mvd_decps_pad = mvd_decps_pad[:, :, 0]

        mD_pred_clipped = torch.clamp_min(mD_pred, min=args.min_depth_pred)

        outputs = model(image1, image2, mD_pred_clipped, intrinsic, posepred, ang_decps_pad, scl_decps_pad, mvd_decps_pad, insmap)
        poseselected = outputs[('afft_all', 2)][0, -1]

        RANSAC_pose_path = os.path.join(args.RANSACPose_root, seq[0:10], seq + "_sync", 'image_02', "{}.pickle".format(str(frmid).zfill(10)))
        RANSAC_pose = pickle.load(open(RANSAC_pose_path, "rb"))

        poseselected_np = poseselected.cpu().numpy()
        pose_bs_np = RANSAC_pose @ np.linalg.inv(RANSAC_pose[0]) @ poseselected_np
        pose_bs_np[0] = poseselected_np

        with open(export_path, 'wb') as handle:
            pickle.dump(pose_bs_np, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return

def get_odomentries(args):
    import glob
    odomentries = list()
    odomseqs = [
        '2011_09_30/2011_09_30_drive_0033_sync',
        '2011_09_30/2011_09_30_drive_0034_sync'
    ]
    # odomseqs = \
    # ['2011_09_30/2011_09_30_drive_0016_sync']
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
    if args.odom_only:
        tot_entries = odom_entries + evaluation_entries
    else:
        tot_entries = odom_entries + entries_expand + evaluation_entries
    tot_entries = list(set(tot_entries))
    tot_entries.sort()

    # export_root = get_export_name(args)
    # ungenerated_entries = list()
    # for entry in tot_entries:
    #     seq, frmidx, _ = entry.split(' ')
    #     if not os.path.exists(os.path.join(export_root, seq, "image_02/{}.pickle".format(str(frmidx).zfill(10)))):
    #         ungenerated_entries.append(entry)
    # return ungenerated_entries
    return odom_entries

def read_odomeval_splits():
    seqmapping = \
    ["09 2011_09_30_drive_0033 000000 001590",
     "10 2011_09_30_drive_0034 000000 001200"]

    # seqmapping = \
    # ["04 2011_09_30_drive_0016 000000 000270"]

    entries = list()
    seqmap = dict()
    for seqm in seqmapping:
        mapentry = dict()
        mapid, seqname, stid, enid = seqm.split(' ')
        mapentry['mapid'] = int(mapid)
        mapentry['stid'] = int(stid)
        mapentry['enid'] = int(enid)
        seqmap[seqname] = mapentry

        for k in range(int(stid), int(enid)):
            entries.append("{}/{}_sync {} {}".format(seqname[0:10], seqname, str(k).zfill(10), 'l'))
    entries.sort()
    return entries, seqmap

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

    print("=> loading checkpoint '{}'".format(args.restore_ckpt))
    loc = 'cuda:{}'.format(args.gpu)
    checkpoint = torch.load(args.restore_ckpt, map_location=loc)
    model.load_state_dict(checkpoint, strict=False)

    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'eppflownet/pose_bin_odom{}.pickle'.format(str(int(32 / args.num_angs)))), 'rb') as f:
        linlogdedge = pickle.load(f)
    minidx = np.argmin(np.abs(linlogdedge))
    print("Min index is :%d, val: %f" % (minidx, linlogdedge[minidx]))

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
    subfoldername = args.restore_ckpt.split('/')[-1].split('.')[0]
    export_folder = os.path.join(args.export_root, exportname, subfoldername)
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

    parser.add_argument('--num_scales', type=int, default=32)
    parser.add_argument('--num_angs', type=int, default=1)

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
    parser.add_argument('--odom_only', action="store_true")
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

    best_performance = 1e10
    best_err_rec = None
    best_pth_name = None

    folds_to_gen = glob.glob(os.path.join(args.RANSACPose_root, '*/'))
    export_root = args.export_root
    for fold_to_gen in folds_to_gen:
        args.RANSACPose_root = fold_to_gen
        subfold_name = fold_to_gen.split('/')[-2]
        args.export_root = os.path.join(export_root, subfold_name)
        if args.distributed:
            args.world_size = ngpus_per_node
            mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            train(args.gpu, ngpus_per_node, args)

        valentries, seqmap = read_odomeval_splits()
        tot_err = dict()
        tot_err['positions_pred'] = 0
        tot_err['positions_RANSAC'] = 0
        tot_err['positions_RANSAC_Odomscale'] = 0

        for s in seqmap.keys():
            posrec = dict()

            pred_pose_root = args.export_root
            pred_poses = list()
            for k in range(int(seqmap[s]['stid']), int(seqmap[s]['enid'])):
                pred_pose_path = os.path.join(pred_pose_root, s[0:10], s + "_sync", 'image_02', "{}.pickle".format(str(k).zfill(10)))
                pred_pose = pickle.load(open(pred_pose_path, "rb"))
                pred_poses.append(pred_pose[0])

            RANSAC_poses = list()
            for k in range(int(seqmap[s]['stid']), int(seqmap[s]['enid'])):
                RANSAC_pose_path = os.path.join(args.RANSACPose_root, s[0:10], s + "_sync", 'image_02', "{}.pickle".format(str(k).zfill(10)))
                RANSAC_pose = pickle.load(open(RANSAC_pose_path, "rb"))
                RANSAC_poses.append(RANSAC_pose[0])

            gtposes_sourse = readlines(os.path.join(project_rootdir, 'exp_poses/kittiodom_gt/poses', "{}.txt".format(str(seqmap[s]['mapid']).zfill(2))))
            gtposes = list()
            for gtpose_src in gtposes_sourse:
                gtpose = np.eye(4).flatten()
                for numstridx, numstr in enumerate(gtpose_src.split(' ')):
                    gtpose[numstridx] = float(numstr)
                gtpose = np.reshape(gtpose, [4, 4])
                gtposes.append(gtpose)

            relposes = list()
            for k in range(len(gtposes) - 1):
                relposes.append(np.linalg.inv(gtposes[k + 1]) @ gtposes[k])

            calib_dir = os.path.join(args.dataset_root, "{}".format(s[0:10]))
            cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
            velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
            imu2cam = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
            intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)

            positions_odom = list()
            scale_odom = list()
            stpos = np.array([[0, 0, 0, 1]]).T
            accumP = np.eye(4)
            for r in relposes:
                accumP = r @ accumP
                positions_odom.append((np.linalg.inv(extrinsic) @ np.linalg.inv(accumP) @ stpos)[0:3, 0])
                scale_odom.append(np.sqrt(np.sum(r[0:3, 3] ** 2) + 1e-10))
            positions_odom = np.array(positions_odom)
            scale_odom = np.array(scale_odom)

            positions_pred = list()
            scale_pred = list()
            stpos = np.array([[0, 0, 0, 1]]).T
            accumP = np.eye(4)
            for p in pred_poses:
                accumP = p @ accumP
                positions_pred.append((np.linalg.inv(extrinsic) @ np.linalg.inv(accumP) @ stpos)[0:3, 0])
                scale_pred.append(np.sqrt(np.sum(p[0:3, 3] ** 2) + 1e-10))
            positions_pred = np.array(positions_pred)
            scale_pred = np.array(scale_pred)

            positions_RANSAC = list()
            scale_RANSAC = list()
            stpos = np.array([[0, 0, 0, 1]]).T
            accumP = np.eye(4)
            for r in RANSAC_poses:
                accumP = r @ accumP
                positions_RANSAC.append((np.linalg.inv(extrinsic) @ np.linalg.inv(accumP) @ stpos)[0:3, 0])
                scale_RANSAC.append(np.sqrt(np.sum(r[0:3, 3] ** 2) + 1e-10))
            positions_RANSAC = np.array(positions_RANSAC)
            scale_RANSAC = np.array(scale_RANSAC)

            positions_RANSAC_Odomscale = list()
            stpos = np.array([[0, 0, 0, 1]]).T
            accumP = np.eye(4)
            for i, r in enumerate(RANSAC_poses):
                r[0:3, 3] = r[0:3, 3] / np.sqrt(np.sum(r[0:3, 3] ** 2) + 1e-10) * np.sqrt(np.sum(relposes[i][0:3, 3] ** 2) + 1e-10)
                accumP = r @ accumP
                positions_RANSAC_Odomscale.append((np.linalg.inv(extrinsic) @ np.linalg.inv(accumP) @ stpos)[0:3, 0])
            positions_RANSAC_Odomscale = np.array(positions_RANSAC_Odomscale)

            posrec['positions_pred'] = positions_pred
            posrec['positions_RANSAC'] = positions_RANSAC
            posrec['positions_RANSAC_Odomscale'] = positions_RANSAC_Odomscale

            scalerec = dict()
            scalerec['scale_pred'] = scale_pred
            scalerec['scale_RANSAC'] = scale_RANSAC

            print("============= %s ============" % (s))
            print("In total %d images," % positions_odom.shape[0])
            for k in posrec.keys():
                err_odom = np.mean(np.sqrt(np.sum((posrec[k] - positions_odom) ** 2, axis=1)))

                if 'scale_{}'.format(k.split('_')[1]) in scalerec.keys():
                    err_scale = np.mean(np.abs(scalerec['scale_{}'.format(k.split('_')[1])] - scale_odom))
                else:
                    err_scale = np.nan

                tot_err[k] += err_odom * len(pred_poses)
                print("%s, err_odom: %f, err_scale: %f" % (k, err_odom.item(), err_scale.item()))
