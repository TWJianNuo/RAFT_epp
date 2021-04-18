from __future__ import print_function, division
import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

import argparse
import PIL.Image as Image
import numpy as np
import shutil
from core.utils.utils import vls_ins, tensor2disp
import torch
import copy
from exp_kitti_eigen_fixation.dataset_kitti_eigen_fixation import get_pose
import pickle

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kittistereo15_root', default='raft', help="name your experiment")
    parser.add_argument('--kittisemantics_root', default='raft', help="name your experiment")
    parser.add_argument('--kittiraw_root', default='raft', help="name your experiment")
    parser.add_argument('--export_root', help="determines which dataset to use for training")

    args = parser.parse_args()

    mappings = [x.rstrip('\n') for x in open(os.path.join(args.kittistereo15_root, 'train_mapping.txt'), 'r')]
    raw_export_root = os.path.join(args.export_root, 'raw')
    datename = 'kittistereo15'
    for k in range(200):
        seqname = "{}_{}".format(datename, str(k).zfill(6))
        export_rgb_folder = os.path.join(raw_export_root, seqname, "{}_sync".format(seqname), 'image_02/data')
        os.makedirs(export_rgb_folder, exist_ok=True)

        for m in range(-10, 11, 1):
            tmpidx = m + 10
            if tmpidx == 10 or tmpidx == 11:
                srcpath_rgb = os.path.join(args.kittistereo15_root, 'data_scene_flow/training/image_2', "{}_{}.png".format(str(k).zfill(6), str(tmpidx)))
                dstpath_rgb = os.path.join(export_rgb_folder, "{}.png".format(str(tmpidx).zfill(10)))
            else:
                srcpath_rgb = os.path.join(args.kittistereo15_root, 'data_scene_flow_multiview/training/image_2', "{}_{}.png".format(str(k).zfill(6), str(tmpidx).zfill(2)))
                dstpath_rgb = os.path.join(export_rgb_folder, "{}.png".format(str(tmpidx).zfill(10)))

            shutil.copy(srcpath_rgb, dstpath_rgb)

        for fold in ['calib_cam_to_cam', 'calib_imu_to_velo', 'calib_velo_to_cam']:
            srcpath_calib = os.path.join(args.kittistereo15_root, 'data_scene_flow_calib/training/{}'.format(fold), "{}.txt".format(str(k).zfill(6)))
            dstpath_calib = os.path.join(raw_export_root, seqname, '{}.txt'.format(fold))
            shutil.copy(srcpath_calib, dstpath_calib)

        m = mappings[k]
        if len(m) > 1:
            cam2cam = read_calib_file(os.path.join(raw_export_root, seqname, 'calib_cam_to_cam.txt'))
            velo2cam = read_calib_file(os.path.join(raw_export_root, seqname, 'calib_velo_to_cam.txt'))
            imu2cam = read_calib_file(os.path.join(raw_export_root, seqname, 'calib_imu_to_velo.txt'))
            intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)
            date, seq, frmidx = m.split(' ')
            rel_pose = get_pose(root=args.kittiraw_root, seq="{}/{}".format(date, seq), index=int(frmidx), extrinsic=extrinsic)

            selfpose_fold = os.path.join(raw_export_root, seqname, "{}_sync".format(seqname), 'relpose/data')
            os.makedirs(selfpose_fold, exist_ok=True)
            selfpose_path = os.path.join(selfpose_fold, '{}.pickle'.format("10".zfill(10)))

            wrthdler = open(selfpose_path, "wb")
            pickle.dump(rel_pose, wrthdler)
            wrthdler.close()

    ## ============================================= ##
    instance_export_root = os.path.join(args.export_root, 'instance')
    datename = 'kittistereo15'
    for k in range(200):
        seqname = "{}_{}".format(datename, str(k).zfill(6))
        export_ins_folder = os.path.join(instance_export_root, seqname, "{}_sync".format(seqname), 'insmap/image_02')
        os.makedirs(export_ins_folder, exist_ok=True)

        srcpath_ins = os.path.join(args.kittisemantics_root, 'training/instance', "{}_10.png".format(str(k).zfill(6)))
        dstpath_ins = os.path.join(export_ins_folder, "{}.png".format(str("10").zfill(10)))

        inslabel = np.array(Image.open(srcpath_ins))
        instance_gt = inslabel % 256
        semantic_gt = inslabel // 256

        instance_organized = np.zeros_like(inslabel)
        inscount = 1
        for sl in np.unique(semantic_gt):
            for il in np.unique(instance_gt[semantic_gt == sl]):
                if il == 0:
                    continue
                selector = (instance_gt == il) * (semantic_gt == sl)
                instance_organized[selector] = inscount
                inscount += 1

        # rgb = np.array(Image.open(os.path.join(args.kittisemantics_root, 'training/image_2', "{}_10.png".format(str(k).zfill(6)))))
        # vls_ins(rgb=rgb, anno=instance_organized)

        Image.fromarray(instance_organized).save(dstpath_ins)

    ## ============================================= ##
    depth_export_root = os.path.join(args.export_root, 'depth')
    datename = 'kittistereo15'
    for k in range(200):
        seqname = "{}_{}".format(datename, str(k).zfill(6))
        export_depth_folder = os.path.join(depth_export_root, seqname, "{}_sync".format(seqname), 'image_02')
        os.makedirs(export_depth_folder, exist_ok=True)

        srcpath_disp = os.path.join(args.kittistereo15_root, 'data_scene_flow/training/disp_occ_0', "{}_10.png".format(str(k).zfill(6)))
        dstpath_depth = os.path.join(export_depth_folder, "{}.png".format("10".zfill(10)))

        disp = np.array(Image.open(srcpath_disp)).astype(np.float32) / 256.0

        cam2cam = read_calib_file(os.path.join(args.kittistereo15_root, 'data_scene_flow_calib/training/calib_cam_to_cam', '{}.txt'.format(str(k).zfill(6))))
        intrinsic = np.eye(4)
        intrinsic[0:3, 0:3] = cam2cam['P_rect_02'].reshape(3, 4)[0:3, 0:3]

        depth = np.zeros_like(disp)
        depth[disp > 0] = intrinsic[0, 0] * 0.54 / disp[disp > 0]
        depth = depth * 256.0

        # depthvls = torch.from_numpy(copy.deepcopy(depth)).unsqueeze(0).unsqueeze(0)
        # depthvls[depthvls == 0] = 1e10
        # tensor2disp(1 / depthvls, vmax=0.15, viewind=0).show()

        Image.fromarray(depth.astype(np.uint16)).save(dstpath_depth)

    ## ============================================= ##
    flow_export_root = os.path.join(args.export_root, 'flow')
    datename = 'kittistereo15'
    for k in range(200):
        seqname = "{}_{}".format(datename, str(k).zfill(6))
        export_flow_folder = os.path.join(flow_export_root, seqname, "{}_sync".format(seqname), 'image_02')
        os.makedirs(export_flow_folder, exist_ok=True)

        srcpath_flow = os.path.join(args.kittistereo15_root, 'data_scene_flow/training/flow_occ', "{}_10.png".format(str(k).zfill(6)))
        dstpath_flow = os.path.join(export_flow_folder, "{}.png".format("10".zfill(10)))
        shutil.copy(srcpath_flow, dstpath_flow)

    srcpath_mapping = os.path.join(args.kittistereo15_root, 'train_mapping.txt')
    dstpath_mapping = os.path.join(args.export_root, 'train_mapping.txt')

    shutil.copy(srcpath_mapping, dstpath_mapping)