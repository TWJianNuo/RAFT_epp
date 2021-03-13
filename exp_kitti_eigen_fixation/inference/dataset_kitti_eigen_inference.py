from __future__ import print_function, division
import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import copy

import os
import math
import random
from glob import glob
import os.path as osp
import PIL.Image as Image

from utils import frame_utils, vls_ins
from torchvision.transforms import ColorJitter
from core.utils.semantic_labels import Label
import time
import cv2

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

class KITTI_eigen(data.Dataset):
    def __init__(self, entries, inheight, inwidth, root='datasets/KITTI', odom_root=None, ins_root=None):
        super(KITTI_eigen, self).__init__()
        self.root = root
        self.odom_root = odom_root
        self.ins_root = ins_root
        self.inheight = inheight
        self.inwidth = inwidth

        self.image_list = list()
        self.intrinsic_list = list()
        self.inspred_list = list()

        self.entries = list()

        for entry in self.remove_dup(entries):
            seq, index = entry.split(' ')
            index = int(index)

            if os.path.exists(os.path.join(root, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))):
                tmproot = root
            else:
                tmproot = odom_root

            img1path = os.path.join(tmproot, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))
            img2path = os.path.join(tmproot, seq, 'image_02', 'data', "{}.png".format(str(index + 1).zfill(10)))

            # Load Intrinsic for each frame
            calib_dir = os.path.join(tmproot, seq.split('/')[0])

            cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
            velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
            imu2cam = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
            intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)
            inspath = os.path.join(self.ins_root, seq, 'insmap/image_02', "{}.png".format(str(index).zfill(10)))

            if not os.path.exists(inspath):
                raise Exception("instance file %s missing" % inspath)

            if not os.path.exists(img2path):
                self.image_list.append([img1path, img1path])
            else:
                self.image_list.append([img1path, img2path])

            self.intrinsic_list.append(intrinsic)
            self.inspred_list.append(inspath)
            self.entries.append(entry)

        assert len(self.intrinsic_list) == len(self.inspred_list) == len(self.entries) == len(self.image_list)

    def remove_dup(self, entries):
        dupentry = list()
        for entry in entries:
            seq, index, _ = entry.split(' ')
            dupentry.append("{} {}".format(seq, index.zfill(10)))

        removed = list(set(dupentry))
        removed.sort()
        return removed

    def __getitem__(self, index):
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        intrinsic = copy.deepcopy(self.intrinsic_list[index])
        inspred = np.array(Image.open(self.inspred_list[index])).astype(np.int)

        data_blob = self.wrapup(img1=img1, img2=img2, intrinsic=intrinsic, insmap=inspred, tag=self.entries[index])
        return data_blob

    def wrapup(self, img1, img2, intrinsic, insmap, tag):
        img1 = torch.from_numpy(img1).permute([2, 0, 1]).float()
        img2 = torch.from_numpy(img2).permute([2, 0, 1]).float()
        intrinsic = torch.from_numpy(intrinsic).float()
        insmap = torch.from_numpy(insmap).unsqueeze(0).int()

        data_blob = dict()
        data_blob['img1'] = img1
        data_blob['img2'] = img2
        data_blob['intrinsic'] = intrinsic
        data_blob['insmap'] = insmap
        data_blob['tag'] = tag

        return data_blob

    def __len__(self):
        return len(self.entries)
