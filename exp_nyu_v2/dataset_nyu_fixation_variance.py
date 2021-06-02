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

from core.utils.utils import vls_ins
from core.utils import frame_utils
from torchvision.transforms import ColorJitter
from core.utils.semantic_labels import Label
import time
import cv2
import pickle
import glob
from core.utils.frame_utils import readFlowKITTI

def latlonToMercator(lat,lon,scale):
    er = 6378137
    mx = scale * lon * np.pi * er / 180
    my = scale * er * np.log(np.tan((90 + lat) * np.pi / 360))
    return mx, my

def latToScale(lat):
    scale = np.cos(lat * np.pi / 180.0)
    return scale

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

def read_into_numbers(path):
    numbs = list()
    with open(path, 'r') as f:
        numstr = f.readlines()[0].rstrip().split(' ')
        for n in numstr:
            numbs.append(float(n))
    return numbs

def rot_from_axisangle(angs):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    rotx = np.eye(3)
    roty = np.eye(3)
    rotz = np.eye(3)

    rotx[1, 1] = np.cos(angs[0])
    rotx[1, 2] = -np.sin(angs[0])
    rotx[2, 1] = np.sin(angs[0])
    rotx[2, 2] = np.cos(angs[0])

    roty[0, 0] = np.cos(angs[1])
    roty[0, 2] = np.sin(angs[1])
    roty[2, 0] = -np.sin(angs[1])
    roty[2, 2] = np.cos(angs[1])

    rotz[0, 0] = np.cos(angs[2])
    rotz[0, 1] = -np.sin(angs[2])
    rotz[1, 0] = np.sin(angs[2])
    rotz[1, 1] = np.cos(angs[2])

    rot = rotz @ (roty @ rotx)
    return rot

def get_pose(root, seq, index, extrinsic):
    scale = latToScale(read_into_numbers(os.path.join(root, seq, 'oxts/data', "{}.txt".format(str(0).zfill(10))))[0])

    # Pose 1
    oxts_path = os.path.join(root, seq, 'oxts/data', "{}.txt".format(str(index).zfill(10)))
    nums = read_into_numbers(oxts_path)
    mx, my = latlonToMercator(nums[0], nums[1], scale)

    pose1 = np.eye(4)
    t1 = np.array([mx, my, nums[2]])
    ang1 = np.array(nums[3:6])

    pose1[0:3, 3] = t1
    pose1[0:3, 0:3] = rot_from_axisangle(ang1)

    # Pose 2
    oxts_path = os.path.join(root, seq, 'oxts/data', "{}.txt".format(str(index + 1).zfill(10)))
    nums = read_into_numbers(oxts_path)
    mx, my = latlonToMercator(nums[0], nums[1], scale)

    pose2 = np.eye(4)
    t2 = np.array([mx, my, nums[2]])
    ang2 = np.array(nums[3:6])

    pose2[0:3, 3] = t2
    pose2[0:3, 0:3] = rot_from_axisangle(ang2)

    rel_pose = np.linalg.inv(pose2 @ np.linalg.inv(extrinsic)) @ (pose1 @ np.linalg.inv(extrinsic))
    return rel_pose.astype(np.float32)

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

def readlines(filename):
    with open(filename, 'r') as f:
        filenames = f.readlines()
    return filenames

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

class NYUV2(data.Dataset):
    def __init__(self, entries, inheight, inwidth, root, istrain=True, RANSACPose_root=None, mdPred_root=None, flowpred_root=None, ban_static_in_trianing=False):
        super(NYUV2, self).__init__()
        self.istrain = istrain
        self.root = root
        self.RANSACPose_root = RANSACPose_root
        self.flowpred_root = flowpred_root
        self.mdPred_root = mdPred_root
        self.inheight = inheight
        self.inwidth = inwidth

        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25/3.14)
        self.asymmetric_color_aug_prob = 0.2

        self.image_list = list()
        self.depth_list = list()
        self.mdDepthpath_list = list()
        self.RANSACPose_list = list()
        self.flowpred_list = list()

        self.entries = list()
        self.nposs = len(glob.glob(os.path.join(mdPred_root, '*/')))

        fx_rgb = 5.1885790117450188e+02
        fy_rgb = 5.1946961112127485e+02
        cx_rgb = 3.2558244941119034e+02
        cy_rgb = 2.5373616633400465e+02
        self.intrinsic = np.eye(4)
        self.intrinsic[0, 0] = fx_rgb
        self.intrinsic[1, 1] = fy_rgb
        self.intrinsic[0, 2] = cx_rgb
        self.intrinsic[1, 2] = cy_rgb

        for entry in entries:
            seq, index = entry.split(' ')
            index = int(index)

            img1path = os.path.join(self.root, seq, 'rgb_{}.png'.format(str(index).zfill(5)))
            img2path = os.path.join(self.root, seq, 'rgb_{}.png'.format(str(index + 1).zfill(5)))

            if not os.path.exists(img1path):
                img1path = img1path.replace('.png', '.jpg')
                img2path = img2path.replace('.png', '.jpg')

            depthpath = os.path.join(self.root, seq, 'sync_depth_{}.png'.format(str(index).zfill(5)))
            mdDepth_path = os.path.join(self.mdPred_root, "{}", seq, "sync_depth_{}.png".format(str(index).zfill(5)))
            posepath = os.path.join(self.RANSACPose_root, seq, "{}.pickle".format(str(index).zfill(5)))
            flowpredpath = os.path.join(self.flowpred_root, seq, "{}.png".format(str(index).zfill(5)))

            if ban_static_in_trianing:
                posepred = pickle.load(open(posepath.format("000"), "rb"))
                if np.sqrt(np.sum(posepred[0:3, 3] ** 2)) < 1e-2:
                    continue

            if not os.path.exists(img2path):
                self.image_list.append([img1path, img1path])
            else:
                self.image_list.append([img1path, img2path])

            self.entries.append(entry)
            self.depth_list.append(depthpath)
            self.mdDepthpath_list.append(mdDepth_path)
            self.RANSACPose_list.append(posepath)
            self.flowpred_list.append(flowpredpath)

        assert len(self.entries) == len(self.depth_list) == len(self.image_list) == len(self.mdDepthpath_list) == len(self.RANSACPose_list)

    def colorjitter(self, img1, img2):
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def __getitem__(self, index):
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        depth = cv2.imread(self.depth_list[index], -1) / 1000.0
        intrinsic = copy.deepcopy(self.intrinsic)

        if self.istrain:
            np.random.seed(index + int(time.time()))
            samp_rep = int(np.random.choice(list(range(self.nposs)), 1).item())
        else:
            samp_rep = int(0)

        mdDepth_pred = cv2.imread(self.mdDepthpath_list[index].format(str(samp_rep).zfill(3)), -1) / 1000.0
        posepred = pickle.load(open(self.RANSACPose_list[index], "rb"))
        flowpred, _ = frame_utils.readFlowKITTI(self.flowpred_list[index])

        img1, img2, depth, mdDepth_pred, flowpred, intrinsic = self.aug_crop(img1, img2, depth, mdDepth_pred, flowpred, intrinsic)

        if self.istrain:
            img1, img2 = self.colorjitter(img1, img2)

        data_blob = self.wrapup(img1=img1, img2=img2, depthmap=depth, mdDepth_pred=mdDepth_pred, flowpred=flowpred, intrinsic=intrinsic, posepred=posepred, tag=self.entries[index])
        return data_blob

    def aug_crop(self, img1, img2, depthmap, mdDepth_pred, flowpred, intrinsic):
        if img1.ndim == 3:
            h, w, _ = img1.shape
        else:
            h, w = img1.shape

        crph = self.inheight
        crpw = self.inwidth

        if crph >= h:
            crph = h

        if crpw >= w:
            crpw = w

        if self.istrain:
            top = np.random.randint(0, h - crph - 1, 1).item()
            left = np.random.randint(0, w - crpw - 1, 1).item()
        else:
            top = int((h - crph) / 2)
            left = int((w - crpw) / 2)

        if not self.istrain:
            eval_mask = np.zeros(depthmap.shape)
            eval_mask[45:471, 41:601] = 1
            depthmap = depthmap * eval_mask

        intrinsic[0, 2] -= left
        intrinsic[1, 2] -= top

        img1 = self.crop_img(img1, left=left, top=top, crph=crph, crpw=crpw)
        img2 = self.crop_img(img2, left=left, top=top, crph=crph, crpw=crpw)
        depthmap = self.crop_img(depthmap, left=left, top=top, crph=crph, crpw=crpw)
        mdDepth_pred = self.crop_img(mdDepth_pred, left=left, top=top, crph=crph, crpw=crpw)
        flowpred = self.crop_img(flowpred, left=left, top=top, crph=crph, crpw=crpw)

        return img1, img2, depthmap, mdDepth_pred, flowpred, intrinsic

    def crop_img(self, img, left, top, crph, crpw):
        img_cropped = img[top:top+crph, left:left+crpw]
        return img_cropped

    def wrapup(self, img1, img2, depthmap, mdDepth_pred, flowpred, intrinsic, posepred, tag):
        img1 = torch.from_numpy(img1).permute([2, 0, 1]).float()
        img2 = torch.from_numpy(img2).permute([2, 0, 1]).float()
        depthmap = torch.from_numpy(depthmap).unsqueeze(0).float()
        mdDepth_pred = torch.from_numpy(mdDepth_pred).unsqueeze(0).float()
        flowpred = torch.from_numpy(flowpred).permute([2, 0, 1]).float()

        posepred = torch.from_numpy(posepred).float()
        intrinsic = torch.from_numpy(intrinsic).float()

        data_blob = dict()
        data_blob['img1'] = img1
        data_blob['img2'] = img2
        data_blob['depthmap'] = depthmap
        data_blob['mdDepth_pred'] = mdDepth_pred
        data_blob['flowpred'] = flowpred
        data_blob['intrinsic'] = intrinsic
        data_blob['posepred'] = posepred
        data_blob['tag'] = tag

        return data_blob

    def __len__(self):
        return len(self.entries)