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

import torch
from exp_kitti_eigen_fixation.dataset_kitti_eigen_fixation import KITTI_eigen, read_deepv2d_pose
from tqdm import tqdm

def readlines(filename):
    with open(filename, 'r') as f:
        filenames = f.readlines()
    return filenames

def validate_RANSAC_odom_offline_accum_absl(args, seqmap, entries):
    import glob
    poses_roots = glob.glob(os.path.join(args.RANSAC_pose_root, '*/'))
    posenames = list()
    for poses_root in poses_roots:
        posenames.append(poses_root.split('/')[-2])

    accumerr = dict()
    pos_recs = dict()
    for val_id, entry in enumerate(tqdm(entries)):
        # Read Pose gt
        seq, frameidx, _ = entry.split(' ')
        seq = seq.split('/')[1]
        if seq not in accumerr.keys():
            accumerr_seq = {'pose_deepv2d': list()}
            accumpos = {'pose_deepv2d': np.array([[0, 0, 0, 1]]).T, 'pose_gt': np.array([[0, 0, 0, 1]]).T}
            pos_recs_seq = {'pose_deepv2d': list(), 'pose_gt': list()}
            for posename in posenames:
                pos_recs_seq[posename] = list()
                accumerr_seq[posename] = list()
                accumpos[posename] = np.array([[0, 0, 0, 1]]).T
            accumerr[seq] = accumerr_seq
            pos_recs[seq] = pos_recs_seq

        frameidx = int(frameidx)
        gtposes_sourse = readlines(os.path.join(args.odomPose_root, "{}.txt".format(str(seqmap[seq[0:21]]['mapid']).zfill(2))))
        if frameidx - int(seqmap[seq[0:21]]['stid']) < 0 or \
                frameidx + 1 - int(seqmap[seq[0:21]]['stid']) < 0 or \
                frameidx - int(seqmap[seq[0:21]]['stid']) >= len(gtposes_sourse) or \
                frameidx + 1 - int(seqmap[seq[0:21]]['stid']) >= len(gtposes_sourse):
            continue
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
        scalegt = np.sqrt(np.sum(posegt[0:3, 3] ** 2)).item()

        pose_deepv2d_path = os.path.join(args.deepv2dpred_root, entry.split(' ')[0], 'posepred', str(frameidx).zfill(10) + '.txt')
        posepred_deepv2d = read_deepv2d_pose(pose_deepv2d_path)
        posepred_deepv2d[0:3, 3:4] = posepred_deepv2d[0:3, 3:4] / np.sqrt(np.sum(posepred_deepv2d[0:3, 3:4] ** 2)) * scalegt

        poses = dict()
        poses['pose_deepv2d'] = posepred_deepv2d
        poses['pose_gt'] = posegt

        for posename in posenames:
            pose_RANSAC_path = os.path.join(args.RANSAC_pose_root, posename, entry.split(' ')[0], 'image_02', str(frameidx).zfill(10) + '.pickle')
            pose_RANSAC_dict = pickle.load(open(pose_RANSAC_path, "rb"))

            pose_RANSAC = np.eye(4)
            pose_RANSAC[0:3, 0:3] = pose_RANSAC_dict['R']
            pose_RANSAC[0:3, 3:4] = pose_RANSAC_dict['t'] * scalegt
            poses[posename] = pose_RANSAC

        # Update Pose
        for k in accumpos.keys():
            accumpos[k] = poses[k] @ accumpos[k]
            pos_recs[seq][k].append(accumpos[k])

        for k in accumerr[seq].keys():
            loss_pos = np.sqrt(np.sum((accumpos[k] - accumpos['pose_gt']) ** 2))
            accumerr[seq][k].append(loss_pos.item())

    accumerr_fin = dict()
    for k in accumerr.keys():
        accumerr_fin[k] = dict()
        for kk in accumerr[k].keys():
            accumerr_fin[k][kk] = np.array(accumerr[k][kk])[-1] / len(accumerr[k][kk])

        tmppos_rec = pos_recs[k]
        plt.figure()
        for kk in tmppos_rec.keys():
            tmppos = tmppos_rec[kk]
            tmppos = np.stack(tmppos, axis=0)
            plt.plot(tmppos[:,0,0], tmppos[:,2,0])
        plt.legend(list(tmppos_rec.keys()))
        plt.axis('scaled')
        plt.savefig(os.path.join(args.RANSAC_pose_root, "{}_gtscale.png".format(k)))
        plt.close()

    weighted_err_dict = dict()
    for k in accumerr_seq.keys():
        totframe = 0
        for s in accumerr_fin.keys():
            totframe += len(accumerr[s][k])

        weighted_err = 0
        for s in accumerr_fin.keys():
            weighted_err += len(accumerr[s][k]) / totframe * accumerr_fin[s][k]

        weighted_err_dict[k] = weighted_err

    for k in weighted_err_dict.keys():
        print("%s : %f" % (k.ljust(50), weighted_err_dict[k]))


def validate_RANSAC_odom_offline_accum_selfscale(args, seqmap, entries):
    import glob
    poses_roots = glob.glob(os.path.join(args.RANSAC_pose_root, '*/'))
    posenames = list()
    for poses_root in poses_roots:
        posenames.append(poses_root.split('/')[-2])
    posenames = posenames[0:1]

    accumerr = dict()
    pos_recs = dict()
    scale_gt = list()
    scale_RANSAC = list()
    scale_deepv2d = list()
    scale_RANSAC_fin = list()
    scale_md_fin = list()
    for val_id, entry in enumerate(tqdm(entries)):
        # Read Pose gt
        seq, frameidx, _ = entry.split(' ')
        seq = seq.split('/')[1]
        if seq not in accumerr.keys():
            accumerr_seq = {'pose_deepv2d': list()}
            accumpos = {'pose_deepv2d': np.array([[0, 0, 0, 1]]).T, 'pose_gt': np.array([[0, 0, 0, 1]]).T}
            pos_recs_seq = {'pose_deepv2d': list(), 'pose_gt': list()}
            for posename in posenames:
                pos_recs_seq[posename] = list()
                accumerr_seq[posename] = list()
                accumpos[posename] = np.array([[0, 0, 0, 1]]).T
            accumerr[seq] = accumerr_seq
            pos_recs[seq] = pos_recs_seq

        frameidx = int(frameidx)
        gtposes_sourse = readlines(os.path.join(args.odomPose_root, "{}.txt".format(str(seqmap[seq[0:21]]['mapid']).zfill(2))))
        if frameidx - int(seqmap[seq[0:21]]['stid']) < 0 or \
                frameidx + 1 - int(seqmap[seq[0:21]]['stid']) < 0 or \
                frameidx - int(seqmap[seq[0:21]]['stid']) >= len(gtposes_sourse) or \
                frameidx + 1 - int(seqmap[seq[0:21]]['stid']) >= len(gtposes_sourse):
            continue
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

        for posename in posenames:
            pose_RANSAC_path = os.path.join(args.RANSAC_pose_root, posename, entry.split(' ')[0], 'image_02', str(frameidx).zfill(10) + '.pickle')
            pose_RANSAC_dict = pickle.load(open(pose_RANSAC_path, "rb"))

            pose_RANSAC = np.eye(4)
            pose_RANSAC[0:3, 0:3] = pose_RANSAC_dict['R']
            pose_RANSAC[0:3, 3:4] = pose_RANSAC_dict['t'] * pose_RANSAC_dict['scale']
            poses[posename] = pose_RANSAC

        # if pose_RANSAC_dict['scale'] == 0:
        #     continue

        scale_gt.append(np.sqrt(np.sum(posegt[0:3, 3] ** 2)))
        scale_RANSAC.append(np.sqrt(np.sum(pose_RANSAC[0:3, 3] ** 2)))
        scale_deepv2d.append(np.sqrt(np.sum(posepred_deepv2d[0:3, 3] ** 2)))

        pose_RANSAC_path = os.path.join(args.RANSAC_finpred_root, entry.split(' ')[0], 'image_02', str(frameidx).zfill(10) + '.pickle')
        pose_RANSAC_dict = pickle.load(open(pose_RANSAC_path, "rb"))
        scale_RANSAC_fin.append(np.sqrt(np.sum(pose_RANSAC_dict[0, 0:3, 3] ** 2)))

        pose_md_path = os.path.join(args.mdPred_root, entry.split(' ')[0], 'image_02/posepred', str(frameidx).zfill(10) + '.pickle')
        pose_md_dict = pickle.load(open(pose_md_path, "rb"))
        scale_md_fin.append(np.sqrt(np.sum(pose_md_dict[0, 0:3, 3] ** 2)))

        # Update Pose
        for k in accumpos.keys():
            accumpos[k] = poses[k] @ accumpos[k]
            pos_recs[seq][k].append(accumpos[k])

        for k in accumerr[seq].keys():
            loss_pos = np.sqrt(np.sum((accumpos[k] - accumpos['pose_gt']) ** 2))
            accumerr[seq][k].append(loss_pos.item())

    diff = np.log(np.array(scale_RANSAC)) - np.log(np.array(scale_gt))
    fig = plt.figure()
    fig.add_subplot(4, 1, 1)
    plt.hist(diff, bins=200, range=(-3, 3))
    plt.title("RANSAC Pose Scale Diff, mean: %f" % (np.mean(diff[np.abs(diff) < 3])))
    fig.add_subplot(4, 1, 2)
    diff = np.log(np.array(scale_deepv2d)) - np.log(np.array(scale_gt))
    plt.hist(diff, bins=200, range=(-3, 3))
    plt.title("Deepv2d Pose Scale Diff, mean: %f" % (np.mean(diff[np.abs(diff) < 3])))
    fig.add_subplot(4, 1, 3)
    diff = np.log(np.array(scale_RANSAC_fin)) - np.log(np.array(scale_gt))
    plt.hist(diff, bins=200, range=(-3, 3))
    plt.title("RANSAC fin Pose Scale Diff, mean: %f" % (np.mean(diff[np.abs(diff) < 3])))
    fig.add_subplot(4, 1, 4)
    diff = np.log(np.array(scale_md_fin)) - np.log(np.array(scale_gt))
    plt.hist(diff, bins=200, range=(-3, 3))
    plt.title("mD fin Pose Scale Diff, mean: %f" % (np.mean(diff[np.abs(diff) < 3])))
    plt.show()

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
    parser.add_argument('--RANSAC_finpred_root', type=str)
    parser.add_argument('--mdPred_root', type=str)

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    train(args)