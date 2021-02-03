from __future__ import print_function, division
import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import sys
sys.path.append('../core')

import os
import math
import random
from glob import glob
import os.path as osp


from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
            self.augmentor.eraser_aug_prob = 0
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        valid = None
        flow = frame_utils.readFlowVRKitti(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() > 0) & (flow[1].abs() > 0)

        # from core.utils.utils import tensor2disp, tensor2grad
        # tensor2disp(valid.float().unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()
        # tensor2grad(((flow[0] / 1242)).float().unsqueeze(0).unsqueeze(0), pos_bar=0.01, neg_bar=0.01, viewind=0).show()

        return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        raise NotImplementedError

    def __len__(self):
        return len(self.image_list)


class VirtualKITTI2(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI', entries=None):
        super(VirtualKITTI2, self).__init__(aug_params, sparse=False)
        if split == 'testing':
            self.is_test = True
        self.root = root
        self.split = split
        self.entries = entries
        self.get_img_lists()

    def get_img_lists(self):
        # if self.split == 'evaluation':
        #     sceneids = [2]
        #     conds = ['morning']
        # elif self.split == 'training':
        #     sceneids = [1, 6, 18, 20]
        #     conds = ['morning', 'sunset']
        #
        # img_lists = list()
        # for k in sceneids:
        #     for c in conds:
        #         pngs = glob(os.path.join(self.root, "Scene{}".format(str(k).zfill(2)), c, 'frames', 'forwardFlow', 'Camera_0', "*.png"))
        #         list.sort(pngs)
        #         for png in pngs:
        #             pngidx = int(png.split('/')[-1].split('.')[-2].split('_')[-1])
        #             img_lists.append([k, c, pngidx])

        self.flow_list = []
        self.image_list = []

        for entry in self.entries:
            k, c, pngidx = entry.split(' ')
            k = int(k)
            pngidx = int(pngidx)
            flowimg_path = os.path.join(self.root, "Scene{}".format(str(k).zfill(2)), c, 'frames', 'forwardFlow', 'Camera_0', "flow_{}.png".format(str(pngidx).zfill(5)))
            rgb_path1 = os.path.join(self.root, "Scene{}".format(str(k).zfill(2)), c, 'frames', 'rgb', 'Camera_0', "rgb_{}.jpg".format(str(pngidx).zfill(5)))
            rgb_path2 = os.path.join(self.root, "Scene{}".format(str(k).zfill(2)), c, 'frames', 'rgb', 'Camera_0', "rgb_{}.jpg".format(str(pngidx + 1).zfill(5)))

            self.flow_list.append(flowimg_path)
            self.image_list.append([rgb_path1, rgb_path2])