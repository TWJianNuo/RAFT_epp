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

from core.datasets import FlowDataset

class KITTI_eigen(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI_eigen, self).__init__(aug_params, sparse=True)
        self.is_test = False

        root = osp.join(root, 'training')
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

        if split == 'training':
            images1 = images1[0:150]
            images2 = images2[0:150]
            self.flow_list = self.flow_list[0:150]
        elif split == 'evaluation':
            images1 = images1[150:200]
            images2 = images2[150:200]
            self.flow_list = self.flow_list[150:200]

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]