import os
import glob
import shutil
from distutils.dir_util import copy_tree

virtualkitti_root = '/media/shengjie/disk1/data/virtual_kitti'
export_root = '/media/shengjie/disk1/data/virtual_kitti_organized'
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
