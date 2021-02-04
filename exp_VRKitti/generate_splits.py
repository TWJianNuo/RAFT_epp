import os
import glob
import shutil
import PIL.Image as Image
import cv2
from distutils.dir_util import copy_tree
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='raft', help="name your experiment")
parser.add_argument('--split_root', help="determines which dataset to use for training")

args = parser.parse_args()
# dataset_root = '/media/shengjie/disk1/data/virtual_kitti_organized'
# split_root = '/home/shengjie/Documents/supporting_projects/RAFT/exp_VRKitti/splits'
dataset_root = args.dataset_root
split_root = args.split_root


seqnums = [1, 2, 6, 18, 20]
secennames = ['morning', 'sunset']
contents_to_copy = ['depth', 'forwardFlow', 'instanceSegmentation', 'rgb', 'textgt']

brokenImageNum = 0
for split in ['evaluation', 'training']:
    if split == 'evaluation':
        sceneids = [2]
        conds = ['morning']
    elif split == 'training':
        sceneids = [1, 6, 18, 20]
        conds = ['morning', 'sunset']

    img_lists = list()
    for k in sceneids:
        for c in conds:
            pngs = glob.glob(os.path.join(dataset_root, "Scene{}".format(str(k).zfill(2)), c, 'frames', 'forwardFlow', 'Camera_0', "*.png"))
            list.sort(pngs)
            for png in pngs:
                pngidx = int(png.split('/')[-1].split('.')[-2].split('_')[-1])
                img_lists.append([k, c, pngidx])

    valid_img_lists = list()

    for k, c, pngidx in img_lists:
        flowimg_path = os.path.join(dataset_root, "Scene{}".format(str(k).zfill(2)), c, 'frames', 'forwardFlow', 'Camera_0', "flow_{}.png".format(str(pngidx).zfill(5)))
        rgb_path1 = os.path.join(dataset_root, "Scene{}".format(str(k).zfill(2)), c, 'frames', 'rgb', 'Camera_0', "rgb_{}.jpg".format(str(pngidx).zfill(5)))
        rgb_path2 = os.path.join(dataset_root, "Scene{}".format(str(k).zfill(2)), c, 'frames', 'rgb', 'Camera_0', "rgb_{}.jpg".format(str(pngidx + 1).zfill(5)))

        try:
            Image.open(flowimg_path).verify()
            Image.open(rgb_path1).verify()
            Image.open(rgb_path2).verify()
        except:
            brokenImageNum += 1
            continue

        valid_img_lists.append("{} {} {}\n".format(k, c, pngidx))

    import random
    random.shuffle(valid_img_lists)
    with open(os.path.join(split_root, '{}_split.txt'.format(split)), "w") as text_file:
        for s in valid_img_lists:
            text_file.write(s)
    print("Broken Image Num of split %s: %d" % (split, brokenImageNum))