from __future__ import print_function, division
import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import pickle

from torch.utils.data import DataLoader
from exp_kitti_eigen.dataset_kitti_eigen import KITTI_eigen

from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from PIL import Image, ImageDraw
from core.utils.flow_viz import flow_to_image
from core.raft import RAFT
from core.utils.utils import InputPadder, forward_interpolate
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable

from scipy.linalg import null_space

from tqdm import tqdm

MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

def read_kittiinstance(val_id):
    instanceroot = '/home/shengjie/Documents/Data/KittiInstance/kitti_semantics/training/instance'
    instancemap = Image.open(os.path.join(instanceroot, "{}_10.png".format(str(val_id).zfill(6))))
    instancemap = np.array(instancemap).astype(np.uint16)
    instancemap_insid = instancemap % 256

    bckselector = instancemap_insid == 0
    instancemap[bckselector] = 0
    return instancemap

def RANSAC_E_instance(flowmap, intrinsic, xxf, yyf):
    flowx = flowmap[0, yyf, xxf]
    flowy = flowmap[1, yyf, xxf]

    xxfo = xxf + flowx
    yyfo = yyf + flowy

    pts1 = np.stack([xxf, yyf], axis=1)
    pts2 = np.stack([xxfo, yyfo], axis=1)

    samplenum = 6000

    if pts1.shape[0] > samplenum:
        rndidx = np.random.choice(range(pts1.shape[0]), size=samplenum, replace=False)
        pts1 = pts1[rndidx, :]
        pts2 = pts2[rndidx, :]

    E, inliers = cv2.findEssentialMat(pts1, pts2, focal=intrinsic[0,0], pp=(intrinsic[0, 2], intrinsic[1, 2]), method=cv2.RANSAC, prob=0.99, threshold=0.1)

    return E

def get_eppipole(E, intrinsic):
    # epp1 is epipole on image1 as the projection of camera origin of image2
    F = np.linalg.inv(intrinsic).T @ E @ np.linalg.inv(intrinsic)
    epp1 = np.squeeze(null_space(F), axis=1)
    epp1 = epp1[0:2] / epp1[2]
    epp2 = np.squeeze(null_space(F.T), axis=1)
    epp2 = epp2[0:2] / epp2[2]

    return epp1, epp2

def R2ang(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    ang = np.zeros([3])
    ang[0] = np.arctan2(R[2, 1], R[2, 2])
    ang[1] = np.arctan2(-R[2, 0], sy)
    ang[2] = np.arctan2(R[1, 0], R[0, 0])
    return ang

def get_reldepth_binrange(depthnp_relfs):
    binnum = 32

    stpos = 0.001
    edpos = 0.999

    depthnp_relfs_sorted = np.sort(depthnp_relfs)
    numpts = depthnp_relfs_sorted.shape[0]

    samplepos = np.linspace(start=stpos, stop=edpos, num=binnum)
    sampled_depth = depthnp_relfs_sorted[(samplepos * numpts).astype(np.int)]

    return sampled_depth

@torch.no_grad()
def get_eppflow_range(args, eval_loader, evaluation_entries):
    """ Peform validation using the KITTI-2015 (train) split """

    t_norms = list()
    t_dirs = list()
    depthnp_relfs = list()
    angs = list()
    for val_id, batch in enumerate(tqdm(eval_loader)):
        rel_pose = batch['rel_pose']
        depth = batch['depth']

        depthnp = depth.squeeze().numpy()
        rel_posenp = rel_pose.squeeze().numpy()

        t_norm = np.sqrt(np.sum(rel_posenp[0:3, 3] ** 2))

        t_dir = rel_posenp[0:3, 3] / t_norm

        depthnp_relf = t_norm / depthnp[depthnp > 0]

        ang = R2ang(rel_posenp[0:3, 0:3])

        t_norms.append(t_norm)
        if t_norm > 0.2:
            t_dirs.append(t_dir)
        depthnp_relfs.append(depthnp_relf)
        angs.append(ang)

    t_norms = np.array(t_norms)
    t_dirs = np.array(t_dirs)
    depthnp_relfs = np.concatenate(depthnp_relfs)
    angs = np.array(angs)

    sampled_depth = get_reldepth_binrange(depthnp_relfs)
    import pickle
    pickle.dump(sampled_depth, open("/home/shengjie/Documents/supporting_projects/RAFT/EppflowCore/depth_bin.pickle", "wb"))

    vlsroot = '/home/shengjie/Desktop/2021_02/2021_02_25'

    plt.figure()
    plt.hist(t_norms, bins=20)
    plt.title("Kitti movement distance")
    plt.savefig(os.path.join(vlsroot, 'mvdistance.png'))
    plt.close()

    plt.figure()
    plt.hist(depthnp_relfs, bins=100)
    for xc in sampled_depth:
        plt.axvline(x=xc, ymin=-0.1, ymax=0.1, color='r', linestyle='dashed', linewidth=0.8)
    plt.title("Kitti rel depth")
    plt.savefig(os.path.join(vlsroot, 'reldepth.png'))
    plt.close()

    fig = plt.figure(figsize=(10, 9))
    ax1 = fig.add_subplot(3, 1, 1)
    plt.hist(t_dirs[:, 0], 20)
    ax1.title.set_text('kitti self move direction on x')
    ax2 = fig.add_subplot(3, 1, 2)
    plt.hist(t_dirs[:, 1], 20)
    ax2.title.set_text('kitti self move direction on y')
    ax3 = fig.add_subplot(3, 1, 3)
    plt.hist(t_dirs[:, 2], 20)
    ax3.title.set_text('kitti self move direction on z')
    plt.savefig(os.path.join(vlsroot, 'mvdirections.png'))
    plt.close()

    fig = plt.figure(figsize=(10, 9))
    ax1 = fig.add_subplot(3, 1, 1)
    plt.hist(angs[:, 0], 20)
    ax1.title.set_text('kitti self rotation direction on x')
    ax2 = fig.add_subplot(3, 1, 2)
    plt.hist(angs[:, 1], 20)
    ax2.title.set_text('kitti self rotation direction on y')
    ax3 = fig.add_subplot(3, 1, 3)
    plt.hist(angs[:, 2], 20)
    ax3.title.set_text('kitti self rotation direction on z')
    plt.savefig(os.path.join(vlsroot, 'rotirections.png'))
    plt.close()

def read_splits():
    split_root = os.path.join(project_rootdir, 'exp_nyu_v2/splits')
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'nyudepthv2_test_files.txt'), 'r')]
    return evaluation_entries

def train(args):
    _, evaluation_entries = read_splits()
    get_eppflow_range(args, evaluation_entries)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--pred_root', type=str)
    parser.add_argument('--num_workers', type=int, default=12)

    args = parser.parse_args()

    train(args)