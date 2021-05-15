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
def get_eppflow_range(args, evaluation_entries):
    """ Peform validation using the KITTI-2015 (train) split """
    diffrecs = list()
    for val_id, entry in enumerate(tqdm(evaluation_entries)):
        seq, index = entry.split(' ')
        index = int(index)

        dgt_path = os.path.join(args.dataset_root, seq, 'sync_depth_{}.png'.format(str(index).zfill(5)))
        gtDepth = cv2.imread(dgt_path, -1)
        gtDepth = gtDepth.astype(np.float32) / 1000.0

        dpred_path = os.path.join(args.pred_root, seq, 'sync_depth_{}.png'.format(str(index).zfill(5)))
        predDepth = cv2.imread(dpred_path, -1)
        predDepth = predDepth.astype(np.float32) / 1000.0

        valid_mask = np.logical_and(gtDepth > args.min_depth_eval, gtDepth < args.max_depth_eval)
        eval_mask = np.zeros(valid_mask.shape)
        eval_mask[45:471, 41:601] = 1
        eval_mask = eval_mask * valid_mask
        eval_mask = eval_mask == 1

        diffrecs += (np.log(predDepth[eval_mask]) - np.log(gtDepth[eval_mask])).tolist()

        # from core.utils.utils import tensor2disp
        # tensor2disp(1 / preddepth, vmax=0.15, viewind=0).show()

    diffrecs = np.array(diffrecs)
    diffrecs_abs = np.sort(np.abs(diffrecs))
    diffrecs_abs = diffrecs_abs[diffrecs_abs < 0.6]
    num_sample = diffrecs_abs.shape[0]
    samplepose = np.linspace(0, int(num_sample * 0.999), 16).astype(np.int)

    sampled_edge = diffrecs_abs[samplepose]
    sampled_edge = np.sort(np.concatenate([sampled_edge[1::], -sampled_edge[1::], np.array([0]), np.array([0.7])]))

    import matplotlib.pyplot as plt
    plt.hist(diffrecs, bins=100)
    for xc in sampled_edge:
        plt.axvline(x=xc, ymin=-0.1, ymax=0.1, color='r', linestyle='dashed', linewidth=0.8)
    plt.title('Log Difference with gt normlaized deoth')
    plt.savefig(os.path.join('/home/shengjie/Desktop/2021_03/2021_03_12', 'evaluation difference on log'))
    plt.close()

    import pickle
    with open('/home/shengjie/Documents/supporting_projects/RAFT/exp_nyu_v2/eppflowenet/depth_bin.pickle', 'wb') as handle:
        pickle.dump(sampled_edge, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def read_splits():
    split_root = os.path.join(project_rootdir, 'exp_nyu_v2/splits')
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'nyudepthv2_test_files.txt'), 'r')]
    return evaluation_entries

def train(args):
    evaluation_entries = read_splits()
    get_eppflow_range(args, evaluation_entries)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--pred_root', type=str)
    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    args = parser.parse_args()

    train(args)