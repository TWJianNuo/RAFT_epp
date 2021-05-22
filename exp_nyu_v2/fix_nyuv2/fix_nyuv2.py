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
parser.add_argument('--nyuv2t_root', type=str)
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


nyuv2_testfold = glob.glob(os.path.join(args.nyuv2_root, '*'))
for fold in nyuv2_testfold:
    if check_name_val(fold.split('/')[-1]):
        shutil.rmtree(fold)
        print("Remove: %s" % fold)


gt_depthmap_root = os.path.join(args.nyuv2t_root, 'nyu_groundtruth.npy')
gt_depths = np.load(gt_depthmap_root)

test_entries = readlines(os.path.join(project_rootdir, 'exp_nyu_v2/splits', 'deepv2d_eigen_test_files.txt'))
for idx, entry in enumerate(test_entries):
    entry = entry.rstrip()
    dirname = os.path.dirname(entry)
    allpngs = glob.glob(os.path.join(args.nyuv2raw_root, dirname))
    img = Image.open(os.path.join(args.nyuv2raw_root, entry))


# imglist = glob.glob('/media/shengjie/disk1/data/nyuv2_raw/basement_0001a/*.ppm')
# imglist.sort()
# for idx, imgpath in enumerate(imglist):
#     rgb = Image.open(imgpath)
#     rgb.save(os.path.join('/media/shengjie/disk1/data/vlsss', str(idx).zfill(3) + '.png'))