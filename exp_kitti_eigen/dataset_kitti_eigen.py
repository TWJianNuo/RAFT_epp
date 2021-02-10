from __future__ import print_function, division
import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp
import PIL.Image as Image

from utils import frame_utils
from torchvision.transforms import ColorJitter
from core.utils.semantic_labels import Label
import time
import cv2
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

@torch.no_grad()
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

class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, depth, semantic_selector, intrinsic, rel_pose):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
            semantic_selector = cv2.resize(semantic_selector, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)

            resizeM = np.eye(4)
            resizeM[0, 0] = scale_x
            resizeM[1, 1] = scale_y
            intrinsic = resizeM @ intrinsic

        if self.do_flip:
            # Depth Seems incorrect
            if np.random.rand() < 0.5:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                depth = depth[:, ::-1]
                semantic_selector = semantic_selector[:, ::-1]

                flipM = np.eye(4)
                flipM[0, 0] = -1
                flipM[0, 2] = img1.shape[1]
                flipM3D = np.linalg.inv(intrinsic) @ flipM @ intrinsic
                rel_pose = flipM3D @ rel_pose @ np.linalg.inv(flipM3D)

                randPose = np.eye(4)
                randPose[0:3, 3] = np.random.randn(3)

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        depth = depth[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        semantic_selector = semantic_selector[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        intrinsic[0, 2] = intrinsic[0, 2] - x0
        intrinsic[1, 2] = intrinsic[1, 2] - y0
        return img1, img2, depth, semantic_selector, intrinsic, rel_pose

    def __call__(self, img1, img2, depth, semantic_selector, intrinsic, rel_pose):
        np.random.seed(int(time.time()))
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, depth, semantic_selector, intrinsic, rel_pose = self.spatial_transform(img1, img2, depth, semantic_selector, intrinsic, rel_pose)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        depth = np.ascontiguousarray(depth)
        semantic_selector = np.ascontiguousarray(semantic_selector)
        valid = (depth > 0) * semantic_selector

        return img1, img2, depth, semantic_selector, intrinsic, rel_pose, valid

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []

        self.depth_list = list()
        self.semantics_list = list()
        self.intrinsic_list = list()
        self.pose_list = list()

    def get_semantic_selector(self, index, h, w):
        semanticspred = Image.open(self.semantics_list[index])
        semanticspred = semanticspred.resize([w, h], Image.NEAREST)
        semanticspred = np.array(semanticspred)
        semantic_selector = np.ones_like(semanticspred)
        for ll in np.unique(semanticspred).tolist():
            if ll in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]:
                semantic_selector[semanticspred == ll] = 0
        return semantic_selector.astype(np.int)

    def get_gt_flow(self, index, depth, valid, intrinsic, rel_pose):
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

        flowgt = np.zeros([h, w, 2])
        flowgt[yyf.astype(np.int), xxf.astype(np.int), 0] = pts2d_oview[0, :] - xxf
        flowgt[yyf.astype(np.int), xxf.astype(np.int), 1] = pts2d_oview[1, :] - yyf
        return flowgt

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        orgw, orgh = img1.size

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        depth = np.array(Image.open(self.depth_list[index])).astype(np.float32) / 256.0
        intrinsic = self.intrinsic_list[index]
        rel_pose = self.pose_list[index]
        semantic_selector = self.get_semantic_selector(index, w=orgw, h=orgh)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, depth, semantic_selector, intrinsic, rel_pose, valid = self.augmentor(img1, img2, depth, semantic_selector, intrinsic, rel_pose)
        else:
            valid = semantic_selector * (depth > 0)

        flow = self.get_gt_flow(index, depth, valid, intrinsic, rel_pose)

        # self.debug(img1, img2, valid, depth, flow, intrinsic, rel_pose, index)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        E = self.getE(rel_pose, intrinsic)

        outputs = dict()
        outputs['img1'] = img1
        outputs['img2'] = img2
        outputs['flow'] = flow
        outputs['valid'] = valid.float()
        outputs['intrinsic'] = torch.from_numpy(intrinsic[0:3, 0:3]).float()
        outputs['rel_pose'] = torch.from_numpy(rel_pose).float()
        outputs['E'] = torch.from_numpy(E).float()
        outputs['semantic_selector'] = torch.from_numpy(semantic_selector).float()
        outputs['depth'] = torch.from_numpy(depth).unsqueeze(0).float()

        return outputs

    def getE(self, rel_pose, intrinsic):
        intrinsic = intrinsic[0:3, 0:3]
        t = rel_pose[0:3, 3] / np.sqrt(np.sum(rel_pose[0:3, 3] ** 2))
        T = self.t2T(t)
        R = rel_pose[0:3, 0:3]

        F = T @ R
        E = np.linalg.inv(intrinsic).T @ F @ np.linalg.inv(intrinsic)
        return E

    def t2T(self, t):
        if torch.is_tensor(t):
            T = torch.zeros([3, 3])
        else:
            T = np.zeros([3, 3])

        T[0, 1] = -t[2]
        T[0, 2] = t[1]
        T[1, 0] = t[2]
        T[1, 2] = -t[0]
        T[2, 0] = -t[1]
        T[2, 1] = t[0]

        return T

    def debug(self, img1, img2, valid, depth, flow, intrinsic, rel_pose, index):
        vlsroot = '/media/shengjie/disk1/visualization/aug_correctness_check'
        h, w, _ = img1.shape
        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        xx = torch.from_numpy(xx).float()
        yy = torch.from_numpy(yy).float()
        selector = torch.from_numpy(valid) == 1

        flowx = flow[selector, 0]
        flowy = flow[selector, 1]

        xxf = xx[selector]
        yyf = yy[selector]
        df = depth[selector]

        xxf_oview = xxf + flowx
        yyf_oview = yyf + flowy

        intrinsic = torch.from_numpy(intrinsic[0:3, 0:3]).float()
        rel_pose = torch.from_numpy(rel_pose).float()
        pts1 = torch.stack([xxf, yyf, torch.ones_like(xxf)], dim=0).float()
        pts2 = torch.stack([xxf_oview, yyf_oview, torch.ones_like(xxf)], dim=0).float()
        R = rel_pose[0:3, 0:3]
        T = self.t2T(rel_pose[0:3, 3] / torch.norm(rel_pose[0:3, 3]))
        F = T @ R
        E = torch.inverse(intrinsic).T @ F @ torch.inverse(intrinsic)
        cons = torch.sum(pts2.T @ E * pts1.T, dim=1).abs().max()

        import matplotlib.pyplot as plt
        cm = plt.get_cmap('magma')
        vmax = 0.15
        tnp = 1 / df / vmax
        tnp = cm(tnp)

        fig = plt.figure(figsize=(16, 9))
        fig.add_subplot(2, 1, 1)
        plt.scatter(xxf.numpy(), yyf.numpy(), 1, tnp)
        plt.imshow(np.array(img1))

        fig.add_subplot(2, 1, 2)
        plt.scatter(xxf_oview.numpy(), yyf_oview.numpy(), 1, tnp)
        plt.imshow(np.array(img2))

        seq, frmidx, _ = self.entries[index].split(' ')
        plt.savefig(os.path.join(vlsroot, "{}_{}.png".format(seq.split('/')[1], str(frmidx).zfill(10))))
        plt.close()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)

class KITTI_eigen(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI', entries=None, semantics_root=None, depth_root=None):
        super(KITTI_eigen, self).__init__(aug_params, sparse=True)
        self.is_test = False
        self.root = root
        self.semantics_root = semantics_root
        self.depth_root = depth_root
        self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

        self.image_list = list()
        self.depth_list = list()
        self.semantics_list = list()
        self.intrinsic_list = list()
        self.pose_list = list()

        self.entries = entries
        for entry in entries:
            seq, index, _ = entry.split(' ')
            index = int(index)

            img1path = os.path.join(root, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))
            img2path = os.path.join(root, seq, 'image_02', 'data', "{}.png".format(str(index + 1).zfill(10)))
            depthpath = os.path.join(depth_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))

            if not (os.path.exists(img2path) and os.path.exists(depthpath)):
                continue

            self.image_list += [[img1path, img2path]]
            self.depth_list.append(depthpath)

            semanticspath = os.path.join(semantics_root, seq, 'semantic_prediction/image_02', "{}.png".format(str(index).zfill(10)))
            self.semantics_list.append(semanticspath)

            # Load Intrinsic for each frame
            calib_dir = os.path.join(root, seq.split('/')[0])

            cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
            velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
            imu2cam = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
            intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)

            # Load pose for each frame
            self.pose_list.append(get_pose(root, seq, index, extrinsic))
            self.intrinsic_list.append(intrinsic)

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

