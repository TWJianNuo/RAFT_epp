import os
import glob
import shutil
import PIL.Image as Image
import cv2
from distutils.dir_util import copy_tree
import numpy as np

virtualkitti_root = '/media/shengjie/disk1/data/virtual_kitti'
export_root = '/media/shengjie/disk1/data/virtual_kitti_organized'
split_root = '/home/shengjie/Documents/supporting_projects/RAFT/exp_VRKitti/splits'

if os.path.exists(export_root):
    shutil.rmtree(export_root)
os.makedirs(export_root, exist_ok=True)

seqnums = [1, 2, 6, 18, 20]
secennames = ['morning', 'sunset']
contents_to_copy = ['depth', 'forwardFlow', 'instanceSegmentation', 'rgb', 'textgt']

for seqnum in seqnums:
    for secenname in secennames:
        for content in contents_to_copy:
            source_fold = os.path.join(virtualkitti_root, "vkitti_2.0.3_{}".format(content), 'Scene{}'.format(str(seqnum).zfill(2)), secenname)
            os.makedirs(os.path.join(export_root, 'Scene{}'.format(str(seqnum).zfill(2)), secenname), exist_ok=True)
            target_fold = os.path.join(export_root, 'Scene{}'.format(str(seqnum).zfill(2)), secenname)
            copy_tree(source_fold, target_fold)
            print("Fold %s finished" % source_fold)

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
            pngs = glob.glob(os.path.join(export_root, "Scene{}".format(str(k).zfill(2)), c, 'frames', 'forwardFlow', 'Camera_0', "*.png"))
            list.sort(pngs)
            for png in pngs:
                pngidx = int(png.split('/')[-1].split('.')[-2].split('_')[-1])
                img_lists.append([k, c, pngidx])

    valid_img_lists = list()

    for k, c, pngidx in img_lists:
        flowimg_path = os.path.join(export_root, "Scene{}".format(str(k).zfill(2)), c, 'frames', 'forwardFlow', 'Camera_0', "flow_{}.png".format(str(pngidx).zfill(5)))
        rgb_path1 = os.path.join(export_root, "Scene{}".format(str(k).zfill(2)), c, 'frames', 'rgb', 'Camera_0', "rgb_{}.jpg".format(str(pngidx).zfill(5)))
        rgb_path2 = os.path.join(export_root, "Scene{}".format(str(k).zfill(2)), c, 'frames', 'rgb', 'Camera_0', "rgb_{}.jpg".format(str(pngidx + 1).zfill(5)))

        try:
            Image.open(flowimg_path).verify()
            Image.open(rgb_path1).verify()
            Image.open(rgb_path2).verify()
        except:
            continue

        valid_img_lists.append("{} {} {}\n".format(k, c, pngidx))

    import random
    random.shuffle(valid_img_lists)
    with open(os.path.join(split_root, '{}_split.txt'.format(split)), "w") as text_file:
        for s in valid_img_lists:
            text_file.write(s)