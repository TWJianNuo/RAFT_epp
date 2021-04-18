from __future__ import print_function, division
import os, sys
project_rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader
from exp_poses.dataset_kitti_stereo15_orged import KITTI_eigen_stereo15
from exp_kitti_eigen_fixation.eppflowenet.EppFlowNet_scratch import EppFlowNet

from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from PIL import Image, ImageDraw
from core.utils.flow_viz import flow_to_image
from core.utils.utils import InputPadder, forward_interpolate, tensor2disp, tensor2rgb, vls_ins
from posenet import Posenet
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable
from exp_kitti_eigen_fixation.dataset_kitti_eigen_fixation import read_calib_file, get_intrinsic_extrinsic, get_pose

from tqdm import tqdm
import pickle


def remove_dup(entries):
    dupentry = list()
    for entry in entries:
        seq, index, _ = entry.split(' ')
        dupentry.append("{} {}".format(seq, index.zfill(10)))
    removed = list(set(dupentry))
    removed.sort()
    return removed

def read_splits():
    split_root = os.path.join(project_rootdir, 'exp_pose_mdepth_kitti_eigen/splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'train_files.txt'), 'r')]
    return train_entries

def rot2ang(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    ang = np.zeros([3])
    ang[0] = np.arctan2(R[2, 1], R[2, 2])
    ang[1] = np.arctan2(-R[2, 0], sy)
    ang[2] = np.arctan2(R[1, 0], R[0, 0])
    return ang

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--pred_root', type=str)

    args = parser.parse_args()

    train_entries = read_splits()
    mvprms = list()
    mvprms_pred = list()
    for entry in remove_dup(train_entries):
        seq, index = entry.split(' ')
        index = int(index)

        calib_dir = os.path.join(args.dataset_root, seq.split('/')[0])
        cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
        velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
        imu2cam = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
        intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)

        img1path = os.path.join(args.dataset_root, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))
        img2path = os.path.join(args.dataset_root, seq, 'image_02', 'data', "{}.png".format(str(index + 1).zfill(10)))

        if not os.path.exists(img2path):
            relpose = np.eye(4)
        else:
            relpose = get_pose(args.dataset_root, seq, index, extrinsic)

        ang = rot2ang(relpose[0:3, 0:3])
        t = relpose[0:3, 3]
        mvprm = np.concatenate([ang, t])
        mvprms.append(mvprm)

        posepred_path = os.path.join(args.pred_root, seq, 'image_02/posepred', "{}.pickle".format(str(index).zfill(10)))
        posepred = pickle.load(open(posepred_path, "rb"))[0]
        ang_pred = rot2ang(posepred[0:3, 0:3])
        t_pred = posepred[0:3, 3]
        mvprm_pred = np.concatenate([ang_pred, t_pred])
        mvprms_pred.append(mvprm_pred)


    mvprms = np.array(mvprms)
    mvprms_meaned = mvprms - np.mean(mvprms, axis=0)

    mvprms_pred = np.array(mvprms_pred)
    mvprms_pred_meaned = mvprms_pred - np.mean(mvprms, axis=0)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(mvprms_meaned)
    reduct = pca.transform(mvprms_meaned)
    reduct_pred = pca.transform(mvprms_pred_meaned)

    rndidx = np.random.randint(0, reduct.shape[0], 1000)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(reduct[:, 0], reduct[:, 1], 0.1)
    plt.quiver(reduct_pred[rndidx, 0], reduct_pred[rndidx, 1], reduct[rndidx, 0], reduct[rndidx, 1], scale=1e2, linewidths=0.0001, width=0.001)
    plt.title("PCA Results of 6DOF Pose")
    plt.xlabel("Xaxis")
    plt.ylabel("Yaxis")

    plt.savefig('/home/shengjie/Desktop/7.png', bbox_inches='tight', pad_inches=0)


