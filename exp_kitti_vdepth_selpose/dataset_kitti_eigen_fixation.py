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

class KITTI_eigen(data.Dataset):
    def __init__(self, entries, inheight, inwidth, maxinsnum, root, ins_root, depth_root, depthvls_root, mdPred_root=None,
                 istrain=True, muteaug=False, isgarg=False, banremovedup=False, RANSACPose_root=None):
        super(KITTI_eigen, self).__init__()
        self.istrain = istrain
        self.isgarg = isgarg
        self.muteaug = muteaug
        self.banremovedup = banremovedup
        self.root = root
        self.depth_root = depth_root
        self.depthvls_root = depthvls_root
        self.mdPred_root = mdPred_root
        self.RANSACPose_root = RANSACPose_root
        self.ins_root = ins_root
        self.inheight = inheight
        self.inwidth = inwidth
        self.maxinsnum = maxinsnum

        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25/3.14)
        self.asymmetric_color_aug_prob = 0.2

        self.image_list = list()
        self.depth_list = list()
        self.depthvls_list = list()
        self.mdDepthpath_list = list()
        self.intrinsic_list = list()
        self.pose_list = list()
        self.inspred_list = list()
        self.predDepthpath_list = list()
        self.predPosepath_list = list()
        self.deepv2dpredpath_list = list()
        self.deepv2dpredpose_list = list()
        self.RANSACPose_list = list()

        self.entries = list()

        for entry in self.remove_dup(entries):
            seq, index = entry.split(' ')
            index = int(index)

            img1path = os.path.join(root, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))
            img2path = os.path.join(root, seq, 'image_02', 'data', "{}.png".format(str(index + 1).zfill(10)))
            depthpath = os.path.join(depth_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))
            depthvlspath = os.path.join(depthvls_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))

            # Load Intrinsic for each frame
            calib_dir = os.path.join(root, seq.split('/')[0])

            cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
            velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
            imu2cam = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
            intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)
            inspath = os.path.join(self.ins_root, seq, 'insmap/image_02', "{}.png".format(str(index).zfill(10)))

            if not os.path.exists(depthpath):
                continue

            if mdPred_root is not None:
                mdDepthpath = os.path.join(mdPred_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))
                if not os.path.exists(mdDepthpath):
                    raise Exception("Prediction file %s missing" % mdDepthpath)
                self.mdDepthpath_list.append(mdDepthpath)

            if not os.path.exists(inspath):
                raise Exception("instance file %s missing" % inspath)
            self.inspred_list.append(inspath)

            if RANSACPose_root is not None:
                RANSACPose_path = os.path.join(RANSACPose_root, seq, 'image_02', "{}.pickle".format(str(index).zfill(10)))
                if not os.path.exists(RANSACPose_path):
                    raise Exception("RANSAC Pose file %s missing" % RANSACPose_path)
                self.RANSACPose_list.append(RANSACPose_path)

            if not os.path.exists(img2path):
                self.image_list.append([img1path, img1path])
                self.pose_list.append(np.eye(4))
            else:
                self.image_list.append([img1path, img2path])
                self.pose_list.append(get_pose(root, seq, int(index), extrinsic))

            self.intrinsic_list.append(intrinsic)
            self.entries.append(entry)
            self.depth_list.append(depthpath)
            self.depthvls_list.append(depthvlspath)

        assert len(self.intrinsic_list) == len(self.entries) == len(self.depth_list) == len(self.image_list) == len(self.pose_list) == len(self.inspred_list)

    def remove_dup(self, entries):
        dupentry = list()
        for entry in entries:
            seq, index, _ = entry.split(' ')
            dupentry.append("{} {}".format(seq, index.zfill(10)))

        if self.banremovedup:
            removed = dupentry
        else:
            removed = list(set(dupentry))
        removed.sort()
        return removed

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

        depth = np.array(Image.open(self.depth_list[index])).astype(np.float32) / 256.0
        depthvls = np.array(Image.open(self.depthvls_list[index])).astype(np.float32) / 256.0
        intrinsic = copy.deepcopy(self.intrinsic_list[index])
        rel_pose = copy.deepcopy(self.pose_list[index])
        inspred = np.array(Image.open(self.inspred_list[index])).astype(np.int)
        posepred = pickle.load(open(self.RANSACPose_list[index], "rb"))

        inspred, posepred = self.pad_clip_ins(insmap=inspred, posepred=posepred)

        if self.mdPred_root is not None:
            mdDepth_pred = np.array(Image.open(self.mdDepthpath_list[index])).astype(np.float32) / 256.0
        else:
            mdDepth_pred = None

        img1, img2, depth, depthvls, mdDepth_pred, inspred, intrinsic = self.aug_crop(img1, img2, depth, depthvls, mdDepth_pred, inspred, intrinsic)

        if self.istrain and not self.muteaug:
            img1, img2 = self.colorjitter(img1, img2)

        data_blob = self.wrapup(img1=img1, img2=img2, depthmap=depth, depthvls=depthvls, mdDepth_pred=mdDepth_pred, intrinsic=intrinsic, insmap=inspred, rel_pose=rel_pose, posepred=posepred, tag=self.entries[index])
        return data_blob

    def pad_clip_ins(self, insmap, posepred):
        # posepred_pad = np.zeros([self.maxinsnum, 4, 4])
        posepred_pad = np.eye(4)
        posepred_pad = np.repeat(np.expand_dims(posepred_pad, axis=0), self.maxinsnum, axis=0)

        if posepred is not None:
            currentins = posepred.shape[0]
            if currentins > self.maxinsnum:
                for k in range(self.maxinsnum, currentins):
                    insmap[insmap == k] = 0
                posepred_pad = posepred[0:self.maxinsnum]
            else:
                posepred_pad[0:currentins] = posepred

        return insmap, posepred_pad

    def wrapup(self, img1, img2, depthmap, depthvls, mdDepth_pred, intrinsic, insmap, rel_pose, posepred, tag):
        img1 = torch.from_numpy(img1).permute([2, 0, 1]).float()
        img2 = torch.from_numpy(img2).permute([2, 0, 1]).float()
        depthmap = torch.from_numpy(depthmap).unsqueeze(0).float()
        depthvls = torch.from_numpy(depthvls).unsqueeze(0).float()

        posepred = torch.from_numpy(posepred).float()
        intrinsic = torch.from_numpy(intrinsic).float()
        rel_pose = torch.from_numpy(rel_pose).float()
        insmap = torch.from_numpy(insmap).unsqueeze(0).int()

        data_blob = dict()
        data_blob['img1'] = img1
        data_blob['img2'] = img2
        data_blob['depthmap'] = depthmap
        data_blob['depthvls'] = depthvls
        data_blob['intrinsic'] = intrinsic
        data_blob['insmap'] = insmap
        data_blob['rel_pose'] = rel_pose
        data_blob['posepred'] = posepred
        data_blob['tag'] = tag

        if mdDepth_pred is not None:
            mdDepth_pred = torch.from_numpy(mdDepth_pred).unsqueeze(0).float()
            data_blob['mdDepth_pred'] = mdDepth_pred

        return data_blob

    def crop_img(self, img, left, top, crph, crpw):
        img_cropped = img[top:top+crph, left:left+crpw]
        return img_cropped

    def aug_crop(self, img1, img2, depthmap, depthvls, mdDepth_pred, instancemap, intrinsic):
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
            left = np.random.randint(0, w - crpw - 1, 1).item()
        else:
            left = int((w - crpw) / 2)
        top = int(h - crph)

        if not self.istrain and self.isgarg:
            crop = np.array([0.40810811 * h, 0.99189189 * h, 0.03594771 * w, 0.96405229 * w]).astype(np.int32)
            crop_mask = np.zeros([h, w])
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            depthmap = depthmap * crop_mask
            assert left < crop[2] and left + crpw > crop[3] and top < crop[0]

        intrinsic[0, 2] -= left
        intrinsic[1, 2] -= top

        img1 = self.crop_img(img1, left=left, top=top, crph=crph, crpw=crpw)
        img2 = self.crop_img(img2, left=left, top=top, crph=crph, crpw=crpw)
        depthmap = self.crop_img(depthmap, left=left, top=top, crph=crph, crpw=crpw)
        depthvls = self.crop_img(depthvls, left=left, top=top, crph=crph, crpw=crpw)
        instancemap = self.crop_img(instancemap, left=left, top=top, crph=crph, crpw=crpw)
        if mdDepth_pred is not None:
            mdDepth_pred = self.crop_img(mdDepth_pred, left=left, top=top, crph=crph, crpw=crpw)

        return img1, img2, depthmap, depthvls, mdDepth_pred, instancemap, intrinsic

    def get_gt_flow(self, depth, valid, intrinsic, rel_pose):
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
        return flowgt

    def __len__(self):
        return len(self.entries)

    def debug_pose(self):
        vlsroot = '/media/shengjie/disk1/visualization/imu_accuracy_vls'
        for k in range(500):
            test_idx = np.random.randint(0, len(self.depth_list), 1)[0]

            img1 = Image.open(self.image_list[test_idx][0])
            img2 = Image.open(self.image_list[test_idx][1])

            depth = np.array(Image.open(self.depth_list[test_idx])).astype(np.float32) / 256.0
            depth = torch.from_numpy(depth)

            h, w = depth.shape

            semanticspred = Image.open(self.semantics_list[test_idx])
            semanticspred = semanticspred.resize([w, h], Image.NEAREST)
            semanticspred = np.array(semanticspred)
            semantic_selector = np.ones_like(semanticspred)
            for ll in np.unique(semanticspred).tolist():
                if ll in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]:
                    semantic_selector[semanticspred == ll] = 0
            semantic_selector = torch.from_numpy(semantic_selector).float()

            intrinsic = torch.from_numpy(self.intrinsic_list[test_idx])
            rel_pose = torch.from_numpy(self.pose_list[test_idx])

            xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
            xx = torch.from_numpy(xx).float()
            yy = torch.from_numpy(yy).float()
            selector = depth > 0

            xxf = xx[selector]
            yyf = yy[selector]
            df = depth[selector]
            pts3d = torch.stack([xxf * df, yyf * df, df, torch.ones_like(df)], dim=0)
            pts3d = torch.inverse(intrinsic) @ pts3d
            pts3d_oview = rel_pose @ pts3d
            pts2d_oview = intrinsic @ pts3d_oview
            pts2d_oview[0, :] = pts2d_oview[0, :] / pts2d_oview[2, :]
            pts2d_oview[1, :] = pts2d_oview[1, :] / pts2d_oview[2, :]

            import matplotlib.pyplot as plt
            cm = plt.get_cmap('magma')
            vmax = 0.15
            tnp = 1 / df.numpy() / vmax
            tnp = cm(tnp)

            fig = plt.figure(figsize=(16, 9))
            fig.add_subplot(2, 1, 1)
            plt.scatter(xxf.numpy(), yyf.numpy(), 1, tnp)
            plt.imshow(img1)

            fig.add_subplot(2, 1, 2)
            plt.scatter(pts2d_oview[0, :].numpy(), pts2d_oview[1, :].numpy(), 1, tnp)
            plt.imshow(img2)

            seq, frmidx, _ = self.entries[test_idx].split(' ')
            plt.savefig(os.path.join(vlsroot, "{}_{}.png".format(seq.split('/')[1], str(frmidx).zfill(10))))
            plt.close()

