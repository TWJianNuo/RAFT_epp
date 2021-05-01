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
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pickle
import copy

import torch
from exp_kitti_eigen_fixation.dataset_kitti_eigen_fixation import KITTI_eigen, read_deepv2d_pose
from tqdm import tqdm

def readlines(filename):
    with open(filename, 'r') as f:
        filenames = f.readlines()
    return filenames

def validate_RANSAC_odom_offline_accum_selfscale(args, seqmap, entries):
    accumerr = {'pose_deepv2d': 0, "pose_RANSAC": 0, 'pose_RANSAC_deevp2dscale':0}

    split_root = os.path.join(project_rootdir, 'exp_pose_mdepth_kitti_eigen/splits')
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'val_files_odom.txt'), 'r')]
    totnum = 0
    for val_id, entry in enumerate(tqdm(entries)):
        # Read Pose gt
        if entry not in evaluation_entries:
            continue

        seq, frameidx, _ = entry.split(' ')
        seq = seq.split('/')[1]

        frameidx = int(frameidx)
        gtposes_sourse = readlines(os.path.join(args.odomPose_root, "{}.txt".format(str(seqmap[seq[0:21]]['mapid']).zfill(2))))
        if frameidx - int(seqmap[seq[0:21]]['stid']) < 0 or \
                frameidx + 1 - int(seqmap[seq[0:21]]['stid']) < 0 or \
                frameidx - int(seqmap[seq[0:21]]['stid']) >= len(gtposes_sourse) or \
                frameidx + 1 - int(seqmap[seq[0:21]]['stid']) >= len(gtposes_sourse):
            posegt = np.eye(4)
        else:
            gtposes_str = [gtposes_sourse[frameidx - int(seqmap[seq[0:21]]['stid'])],
                           gtposes_sourse[frameidx + 1 - int(seqmap[seq[0:21]]['stid'])]]
            gtposes = list()
            for gtposestr in gtposes_str:
                gtpose = np.eye(4).flatten()
                for numstridx, numstr in enumerate(gtposestr.split(' ')):
                    gtpose[numstridx] = float(numstr)
                gtpose = np.reshape(gtpose, [4, 4])
                gtposes.append(gtpose)
            posegt = np.linalg.inv(gtposes[1]) @ gtposes[0]

        pose_deepv2d_path = os.path.join(args.deepv2dpred_root, entry.split(' ')[0], 'posepred', str(frameidx).zfill(10) + '.txt')
        posepred_deepv2d = read_deepv2d_pose(pose_deepv2d_path)
        posepred_deepv2d[0:3, 3:4] = posepred_deepv2d[0:3, 3:4]

        poses = dict()
        poses['pose_deepv2d'] = posepred_deepv2d
        poses['pose_gt'] = posegt

        pose_RANSAC_path = os.path.join(args.RANSAC_pose_root, entry.split(' ')[0], 'image_02', str(frameidx).zfill(10) + '.pickle')
        pose_RANSAC = pickle.load(open(pose_RANSAC_path, "rb"))
        poses['pose_RANSAC'] = pose_RANSAC[0]

        poses['pose_RANSAC_deevp2dscale'] = copy.deepcopy(poses['pose_RANSAC'])
        poses['pose_RANSAC_deevp2dscale'][0:3, 3] = poses['pose_RANSAC_deevp2dscale'][0:3, 3] / np.sqrt(np.sum(pose_RANSAC[0][0:3, 3] ** 2) +1e-8) * np.sqrt(np.sum(posepred_deepv2d[0:3, 3] ** 2) +1e-8)

        for k in accumerr.keys():
            accumerr[k] += np.sum(np.abs(posegt - poses[k]))
        totnum = totnum + 1

    for k in accumerr.keys():
        accumerr[k] = accumerr[k] / totnum
    print("Eval num: %d" % totnum)
    print(accumerr)

def generate_seqmapping():
    seqmapping = \
    ['00 2011_10_03_drive_0027 000000 004540',
     "04 2011_09_30_drive_0016 000000 000270",
     "05 2011_09_30_drive_0018 000000 002760",
     "07 2011_09_30_drive_0027 000000 001100"]

    entries = list()
    seqmap = dict()
    for seqm in seqmapping:
        mapentry = dict()
        mapid, seqname, stid, enid = seqm.split(' ')
        mapentry['mapid'] = int(mapid)
        mapentry['stid'] = int(stid)
        mapentry['enid'] = int(enid)
        seqmap[seqname] = mapentry

        for k in range(int(stid), int(enid)):
            entries.append("{}/{}_sync {} {}".format(seqname[0:10], seqname, str(k).zfill(10), 'l'))

    return seqmap, entries

def train(args):
    seqmap, entries = generate_seqmapping()
    entries = entries
    validate_RANSAC_odom_offline_accum_selfscale(args, seqmap, entries)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--RANSAC_pose_root', type=str)
    parser.add_argument('--deepv2dpred_root', type=str)
    parser.add_argument('--odomPose_root', type=str)

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    train(args)