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

def generate_seqmapping():
    seqmapping = [
        "09 2011_09_30_drive_0033 000000 001590",
        "10 2011_09_30_drive_0034 000000 001200"]

    seqmap = dict()
    for seqm in seqmapping:
        mapentry = dict()
        mapid, seqname, stid, enid = seqm.split(' ')
        mapentry['mapid'] = int(mapid)
        mapentry['stid'] = int(stid)
        mapentry['enid'] = int(enid)
        seqmap[seqname] = mapentry
    return seqmap

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

class KITTI_odom(data.Dataset):
    def __init__(self, entries, inheight, inwidth, maxinsnum, root, ins_root, depth_root=None, depthvls_root=None, mdPred_root=None, num_samples=4, linlogdedge=None,
                 prediction_root=None, istrain=True, muteaug=False, isgarg=False, banremovedup=False, deepv2dpred_root=None, flowPred_root=None, RANSACPose_root=None):
        super(KITTI_odom, self).__init__()
        self.istrain = istrain
        self.isgarg = isgarg
        self.muteaug = muteaug
        self.banremovedup = banremovedup
        self.root = root
        self.depth_root = depth_root
        self.depthvls_root = depthvls_root
        self.mdPred_root = mdPred_root
        self.prediction_root = prediction_root
        self.deepv2dpred_root = deepv2dpred_root
        self.flowPred_root = flowPred_root
        self.RANSACPose_root = RANSACPose_root
        self.ins_root = ins_root
        self.inheight = inheight
        self.inwidth = inwidth
        self.maxinsnum = maxinsnum
        self.num_samples = num_samples
        self.linlogdedge = linlogdedge

        self.npossibility = len(glob(os.path.join(RANSACPose_root, '*')))

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
        self.flowpredpath_list = list()
        self.RANSACPose_list = list()

        seqmap = generate_seqmapping()

        self.odom_gt_relpose_rec = dict()
        for seq in seqmap.keys():
            gtpose_root = os.path.join(project_rootdir, 'exp_poses/kittiodom_gt/poses', str(seqmap[seq]["mapid"]).zfill(2) + '.txt')

            gtposes = list()
            gtposes_sourse = readlines(gtpose_root)
            for gtpose_src in gtposes_sourse:
                gtpose = np.eye(4).flatten()
                for numstridx, numstr in enumerate(gtpose_src.split(' ')):
                    gtpose[numstridx] = float(numstr)
                gtpose = np.reshape(gtpose, [4, 4])
                gtposes.append(gtpose)

            relposes = list()
            for k in range(len(gtposes)):
                if k < len(gtposes) - 1:
                    relposes.append(np.linalg.inv(gtposes[k + 1]) @ gtposes[k])
                else:
                    relposes.append(np.eye(4))
            self.odom_gt_relpose_rec[seq] = relposes

        self.gt_relpose_rec = list()

        self.entries = list()
        for entry in self.remove_dup(entries):
            seq, index = entry.split(' ')
            index = int(index)

            img1path = os.path.join(root, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))
            img2path = os.path.join(root, seq, 'image_02', 'data', "{}.png".format(str(index + 1).zfill(10)))
            depthvlspath = os.path.join(depthvls_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))

            # Load Intrinsic for each frame
            calib_dir = os.path.join(root, seq.split('/')[0])

            cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
            velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
            imu2cam = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
            intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)
            inspath = os.path.join(self.ins_root, seq, 'insmap/image_02', "{}.png".format(str(index).zfill(10)))

            if depth_root is not None:
                depthpath = os.path.join(depth_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))
                if not os.path.exists(depthpath):
                    continue
                self.depth_list.append(depthpath)

            if mdPred_root is not None:
                mdDepthpath = os.path.join(mdPred_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))
                if not os.path.exists(mdDepthpath):
                    raise Exception("Prediction file %s missing" % mdDepthpath)
                self.mdDepthpath_list.append(mdDepthpath)

            if not os.path.exists(inspath):
                raise Exception("instance file %s missing" % inspath)
            self.inspred_list.append(inspath)

            if deepv2dpred_root is not None:
                deepv2d_path = os.path.join(deepv2dpred_root, seq, 'depthpred', "{}.png".format(str(index).zfill(10)))
                deepv2dpose_path = os.path.join(deepv2dpred_root, seq, 'posepred', "{}.txt".format(str(index).zfill(10)))
                if os.path.exists(deepv2d_path):
                    self.deepv2dpredpath_list.append(deepv2d_path)
                    self.deepv2dpredpose_list.append(deepv2dpose_path)
                else:
                    raise Exception("Deepv2d prediction file %s missing" % deepv2d_path)

            if prediction_root is not None:
                predDepthpath = os.path.join(prediction_root, seq, 'image_02/depthpred', "{}.png".format(str(index).zfill(10)))
                predPosepath = os.path.join(prediction_root, seq, 'image_02/posepred', "{}.pickle".format(str(index).zfill(10)))
                if not os.path.exists(predDepthpath) or not os.path.exists(predPosepath):
                    raise Exception("prediction file %s missing" % predDepthpath)
                self.predDepthpath_list.append(predDepthpath)
                self.predPosepath_list.append(predPosepath)

            if RANSACPose_root is not None:
                RANSACPose_path = os.path.join(RANSACPose_root, "{}", seq, 'image_02', "{}.pickle".format(str(index).zfill(10)))
                self.RANSACPose_list.append(RANSACPose_path)

            if flowPred_root is not None:
                flowPred_path = os.path.join(flowPred_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))
                if not os.path.exists(flowPred_path):
                    raise Exception("Prediction file %s missing" % flowPred_path)
                self.flowpredpath_list.append(flowPred_path)

            if not os.path.exists(img2path):
                self.image_list.append([img1path, img1path])
                self.pose_list.append(np.eye(4))
            else:
                self.image_list.append([img1path, img2path])
                self.pose_list.append(get_pose(root, seq, int(index), extrinsic))

            gt_relpose = self.odom_gt_relpose_rec[seq.split('/')[1][0:21]][index - int(seqmap[seq.split('/')[1][0:21]]['stid'])]
            self.gt_relpose_rec.append(gt_relpose)

            self.intrinsic_list.append(intrinsic)
            self.depthvls_list.append(depthvlspath)
            self.entries.append(entry)

        assert len(self.intrinsic_list) == len(self.entries) == len(self.image_list) == len(self.pose_list) == len(self.inspred_list)

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

    def rot2ang(self, R):
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        ang = np.zeros([3])
        ang[0] = np.arctan2(R[2, 1], R[2, 2])
        ang[1] = np.arctan2(-R[2, 0], sy)
        ang[2] = np.arctan2(R[1, 0], R[0, 0])
        return ang

    def __getitem__(self, index):
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        depthvls = np.array(Image.open(self.depthvls_list[index])).astype(np.float32) / 256.0
        intrinsic = copy.deepcopy(self.intrinsic_list[index])
        rel_pose = copy.deepcopy(self.pose_list[index])
        inspred = np.array(Image.open(self.inspred_list[index])).astype(np.int)

        if self.RANSACPose_root is not None:
            np.random.seed(index + int(time.time()))
            if self.num_samples == 1:
                rep_idces = [0]
            elif self.num_samples == 2:
                rep_idces = [0, 1]
            else:
                rep_idces = [0, 1, 2, 3]
            # rep_idces = [0, 0, 0, 0]
            posepred = list()

            ang_decps_l = list()
            scl_decps_l = list()
            mvd_decps_l = list()
            for rep_idx in rep_idces:
                posepred_cur = pickle.load(open(self.RANSACPose_list[index].format(str(rep_idx).zfill(3)), "rb"))
                for l in self.linlogdedge:
                    scaledpose = copy.deepcopy(posepred_cur)
                    for kk in range(scaledpose.shape[0]):
                        tmpscale = np.sqrt(np.sum(scaledpose[kk, 0:3, 3] ** 2) + 1e-12)
                        adjustedscale = np.exp(np.log(tmpscale) + l)
                        scaledpose[kk, 0:3, 3] = scaledpose[kk, 0:3, 3] / tmpscale * adjustedscale
                    posepred.append(scaledpose)

                posepred_decps = copy.deepcopy(posepred_cur)
                ang_decps = np.zeros([posepred_decps.shape[0], 3])
                scl_decps = np.zeros([posepred_decps.shape[0], 1])
                mvd_decps = np.zeros([posepred_decps.shape[0], 3])
                for k in range(posepred_decps.shape[0]):
                    ang_decps[k, :] = self.rot2ang(posepred_decps[k])
                    tmps = np.sum(posepred_decps[k, 0:3, 3] ** 2)
                    if tmps == 0:
                        scl_decps[k, 0] = 1e-6
                        mvd_decps[k, :] = np.array([0, 0, -1])
                    else:
                        scl_decps[k, 0] = np.sqrt(tmps)
                        mvd_decps[k, :] = posepred_decps[k, 0:3, 3] / scl_decps[k, 0]
                ang_decps_l.append(ang_decps)
                scl_decps_l.append(scl_decps)
                mvd_decps_l.append(mvd_decps)

            posepred = np.stack(posepred, axis=0)
            ang_decps = np.stack(ang_decps_l, axis=0)
            scl_decps = np.stack(scl_decps_l, axis=0)
            mvd_decps = np.stack(mvd_decps_l, axis=0)
        else:
            posepred = None
            ang_decps = None
            scl_decps = None
            mvd_decps = None

        relpose_odom_gt = copy.deepcopy(self.gt_relpose_rec[index])

        inspred, posepred, ang_decps_pad, scl_decps_pad, mvd_decps_pad = self.pad_clip_ins(insmap=inspred, posepred=posepred, ang_decps=ang_decps, scl_decps=scl_decps, mvd_decps=mvd_decps)

        if not hasattr(self, 'deepv2dpred_root'):
            self.deepv2dpred_root = None

        if self.depth_root is not None:
            depth = np.array(Image.open(self.depth_list[index])).astype(np.float32) / 256.0
        else:
            depth = None

        if self.deepv2dpred_root is not None:
            depthpred_deepv2d = np.array(Image.open(self.deepv2dpredpath_list[index])).astype(np.float32) / 256.0
            posepred_deepv2d = read_deepv2d_pose(self.deepv2dpredpose_list[index])
        else:
            depthpred_deepv2d = None
            posepred_deepv2d = None

        if self.mdPred_root is not None:
            mdDepth_pred = np.array(Image.open(self.mdDepthpath_list[index])).astype(np.float32) / 256.0
        else:
            mdDepth_pred = None

        if self.flowPred_root is not None:
            flowpred_RAFT, valid_flow = readFlowKITTI(self.flowpredpath_list[index])
        else:
            flowpred_RAFT = None

        img1, img2, depth, depthvls, depthpred_deepv2d, mdDepth_pred, inspred, flowpred_RAFT, intrinsic = self.aug_crop(img1, img2, depth, depthvls, depthpred_deepv2d, mdDepth_pred, inspred, flowpred_RAFT, intrinsic)

        if self.depth_root is not None:
            flowgt = self.get_gt_flow(depth=depth, valid=(inspred==0) * (depth>0), intrinsic=intrinsic, rel_pose=rel_pose)
        else:
            flowgt = None
        flowgt_vls = self.get_gt_flow(depth=depthvls, valid=(inspred==0) * (depthvls>0), intrinsic=intrinsic, rel_pose=rel_pose)
        if self.istrain and not self.muteaug:
            img1, img2 = self.colorjitter(img1, img2)

        data_blob = self.wrapup(img1=img1, img2=img2, flowgt=flowgt, flowgt_vls=flowgt_vls, depthmap=depth, depthvls=depthvls,
                                depthpred_deepv2d=depthpred_deepv2d, mdDepth_pred=mdDepth_pred, intrinsic=intrinsic, insmap=inspred, flowpred=flowpred_RAFT, rel_pose=rel_pose,
                                posepred=posepred, posepred_deepv2d=posepred_deepv2d, relpose_odom_gt=relpose_odom_gt, ang_decps_pad=ang_decps_pad, scl_decps_pad=scl_decps_pad, mvd_decps_pad=mvd_decps_pad, tag=self.entries[index])
        return data_blob

    def pad_clip_ins(self, insmap, posepred, ang_decps, scl_decps, mvd_decps):
        # posepred_pad = np.zeros([self.maxinsnum, 4, 4])
        posepred_pad = np.eye(4)
        posepred_pad = np.repeat(np.expand_dims(posepred_pad, axis=0), self.maxinsnum, axis=0)
        posepred_pad = np.repeat(np.expand_dims(posepred_pad, axis=0), self.num_samples * self.linlogdedge.shape[0], axis=0)

        ang_decps_pad = np.zeros([self.num_samples, self.maxinsnum, 3])
        scl_decps_pad = np.zeros([self.num_samples, self.maxinsnum, 1])
        mvd_decps_pad = np.zeros([self.num_samples, self.maxinsnum, 3])

        if posepred is not None:
            currentins = posepred.shape[1]
            assert currentins == insmap.max() + 1
            if currentins > self.maxinsnum:
                for k in range(self.maxinsnum, currentins):
                    insmap[insmap == k] = 0
                posepred_pad = posepred[:, 0:self.maxinsnum]
                ang_decps_pad = ang_decps[:, 0:self.maxinsnum]
                scl_decps_pad = scl_decps[:, 0:self.maxinsnum]
                mvd_decps_pad = mvd_decps[:, 0:self.maxinsnum]
            else:
                posepred_pad[:, 0:currentins] = posepred
                ang_decps_pad[:, 0:currentins] = ang_decps
                scl_decps_pad[:, 0:currentins] = scl_decps
                mvd_decps_pad[:, 0:currentins] = mvd_decps

        return insmap, posepred_pad, ang_decps_pad, scl_decps_pad, mvd_decps_pad

    def wrapup(self, img1, img2, flowgt, flowgt_vls, depthmap, depthvls, depthpred_deepv2d, mdDepth_pred, intrinsic, insmap, flowpred, rel_pose, posepred, posepred_deepv2d, relpose_odom_gt, ang_decps_pad, scl_decps_pad, mvd_decps_pad, tag):
        img1 = torch.from_numpy(img1).permute([2, 0, 1]).float()
        img2 = torch.from_numpy(img2).permute([2, 0, 1]).float()
        flowgt_vls = torch.from_numpy(flowgt_vls).permute([2, 0, 1]).float()
        depthvls = torch.from_numpy(depthvls).unsqueeze(0).float()
        posepred = torch.from_numpy(posepred).float()
        relpose_odom_gt = torch.from_numpy(relpose_odom_gt).unsqueeze(0).float()
        intrinsic = torch.from_numpy(intrinsic).float()
        rel_pose = torch.from_numpy(rel_pose).float()
        ang_decps_pad = torch.from_numpy(ang_decps_pad).float()
        scl_decps_pad = torch.from_numpy(scl_decps_pad).float()
        mvd_decps_pad = torch.from_numpy(mvd_decps_pad).float()
        insmap = torch.from_numpy(insmap).unsqueeze(0).int()

        data_blob = dict()
        data_blob['img1'] = img1
        data_blob['img2'] = img2
        data_blob['flowgt_vls'] = flowgt_vls
        data_blob['depthvls'] = depthvls
        data_blob['intrinsic'] = intrinsic
        data_blob['insmap'] = insmap
        data_blob['rel_pose'] = rel_pose
        data_blob['relpose_odom_gt'] = relpose_odom_gt
        data_blob['posepred'] = posepred
        data_blob['ang_decps_pad'] = ang_decps_pad
        data_blob['scl_decps_pad'] = scl_decps_pad
        data_blob['mvd_decps_pad'] = mvd_decps_pad
        data_blob['tag'] = tag

        if depthpred_deepv2d is not None:
            depthpred_deepv2d = torch.from_numpy(depthpred_deepv2d).unsqueeze(0).float()
            data_blob['depthpred_deepv2d'] = depthpred_deepv2d

        if mdDepth_pred is not None:
            mdDepth_pred = torch.from_numpy(mdDepth_pred).unsqueeze(0).float()
            data_blob['mdDepth_pred'] = mdDepth_pred

        if depthmap is not None:
            depthmap = torch.from_numpy(depthmap).unsqueeze(0).float()
            data_blob['depthmap'] = depthmap

        if posepred_deepv2d is not None:
            posepred_deepv2d = torch.from_numpy(posepred_deepv2d).unsqueeze(0).float()
            data_blob['posepred_deepv2d'] = posepred_deepv2d

        if flowpred is not None:
            flowpred = torch.from_numpy(flowpred).permute([2, 0, 1]).float()
            data_blob['flowpred'] = flowpred

        if flowgt is not None:
            flowgt = torch.from_numpy(flowgt).permute([2, 0, 1]).float()
            data_blob['flowgt'] = flowgt

        return data_blob

    def crop_img(self, img, left, top, crph, crpw):
        img_cropped = img[top:top+crph, left:left+crpw]
        return img_cropped

    def aug_crop(self, img1, img2, depthmap, depthvls, depthpred_deepv2d, mdDepth_pred, instancemap, flowpred_RAFT, intrinsic):
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

        if not self.istrain and self.isgarg and (depthmap is not None):
            crop = np.array([0.40810811 * h, 0.99189189 * h, 0.03594771 * w, 0.96405229 * w]).astype(np.int32)
            crop_mask = np.zeros([h, w])
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            depthmap = depthmap * crop_mask
            assert left < crop[2] and left + crpw > crop[3] and top < crop[0]

        intrinsic[0, 2] -= left
        intrinsic[1, 2] -= top

        img1 = self.crop_img(img1, left=left, top=top, crph=crph, crpw=crpw)
        img2 = self.crop_img(img2, left=left, top=top, crph=crph, crpw=crpw)
        depthvls = self.crop_img(depthvls, left=left, top=top, crph=crph, crpw=crpw)
        instancemap = self.crop_img(instancemap, left=left, top=top, crph=crph, crpw=crpw)
        if depthpred_deepv2d is not None:
            depthpred_deepv2d = self.crop_img(depthpred_deepv2d, left=left, top=top, crph=crph, crpw=crpw)
        if mdDepth_pred is not None:
            mdDepth_pred = self.crop_img(mdDepth_pred, left=left, top=top, crph=crph, crpw=crpw)
        if flowpred_RAFT is not None:
            flowpred_RAFT = self.crop_img(flowpred_RAFT, left=left, top=top, crph=crph, crpw=crpw)
        if depthmap is not None:
            depthmap = self.crop_img(depthmap, left=left, top=top, crph=crph, crpw=crpw)

        return img1, img2, depthmap, depthvls, depthpred_deepv2d, mdDepth_pred, instancemap, flowpred_RAFT, intrinsic

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

