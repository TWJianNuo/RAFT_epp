import glob
import argparse
import os, sys
import shutil
import numpy as np
import PIL.Image as Image
project_rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

parser = argparse.ArgumentParser()
parser.add_argument('--nyuv2_root', type=str)
parser.add_argument('--nyuv2_organized_root', type=str)
parser.add_argument('--nyuv2raw_root', type=str)
args = parser.parse_args()

def readlines(filename):
    with open(filename, 'r') as f:
        filenames = f.readlines()
    return filenames

def check_name_val(foldname):
    number = list(range(0, 10))
    istest = True
    for n in number:
        if str(n) in foldname:
            istest = False
    return istest


all_scenes_organized = [x.split('/')[-2] for x in glob.glob(os.path.join(args.nyuv2_organized_root, '*/'))]
bts_train_sceens = [x.split('/')[-2] for x in glob.glob(os.path.join(args.nyuv2_root, '*/'))]

for idx, bts_scene in enumerate(bts_train_sceens):
    if bts_scene not in all_scenes_organized:
        src_fold = os.path.join(args.nyuv2_root, bts_scene)
        dst_fold = os.path.join(args.nyuv2_organized_root, bts_scene)
        shutil.copytree(src_fold, dst_fold)
        print("%s copied" % dst_fold)


bts_nyu_train_entries_rec = '/home/shengjie/Documents/supporting_projects/RAFT/exp_nyu_v2/splits/nyudepthv2_train_files.txt'
bts_nyu_train_entries = readlines(bts_nyu_train_entries_rec)
organized_train_scenes = list()
for entry in bts_nyu_train_entries:
    sc, _ = entry.rstrip().split(' ')
    organized_train_scenes.append(sc)

organized_train_scenes = list(set(organized_train_scenes))
organized_train_entries = list()
for sc in organized_train_scenes:
    rgb_entries = glob.glob(os.path.join(args.nyuv2_organized_root, sc, 'rgb_*'))
    rgb_entries.sort()
    for rgben in rgb_entries:
        idx_str = rgben.split('/')[-1].split('.')[0].split('_')[1].zfill(5)
        depth_path = os.path.join(args.nyuv2_organized_root, sc, 'sync_depth_{}.png'.format(idx_str))
        assert os.path.exists(depth_path)
        organized_train_entries.append("{} {}".format(sc, idx_str))
import random
random.shuffle(organized_train_entries)

with open('/home/shengjie/Documents/supporting_projects/RAFT/exp_nyu_v2/splits/nyudepthv2_organized_train_files.txt', 'w') as f:
    for idx, en in enumerate(organized_train_entries):
        if idx == len(organized_train_entries) - 1:
            f.writelines(en)
        else:
            f.writelines(en + '\n')


organized_scenes = [x.split('/')[-2] for x in glob.glob(os.path.join(args.nyuv2_organized_root, '*/'))]
organized_test_scenes = list()
for x in organized_scenes:
    if 'test' in x:
        organized_test_scenes.append(x)

organized_test_entries = list()
for x in organized_test_scenes:
    organized_test_entries.append("{} {}".format(x, str(0).zfill(5)))
with open('/home/shengjie/Documents/supporting_projects/RAFT/exp_nyu_v2/splits/nyudepthv2_organized_test_files.txt', 'w') as f:
    for idx, en in enumerate(organized_test_entries):
        if idx == len(organized_test_entries) - 1:
            f.writelines(en)
        else:
            f.writelines(en + '\n')
