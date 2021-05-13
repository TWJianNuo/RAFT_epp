import os, sys
project_rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from exp_kitti_eigen_fixation.dataset_kitti_eigen_fixation import KITTI_eigen

import core.datasets as datasets
from core.utils import flow_viz, frame_utils
from core.raft import RAFT
from core.utils.utils import InputPadder, forward_interpolate, tensor2rgb
from numpy.random import default_rng
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def get_odomentries(args):
    import glob
    odomentries = list()
    odomseqs = [
        '2011_10_03/2011_10_03_drive_0027_sync',
        '2011_09_30/2011_09_30_drive_0016_sync',
        '2011_09_30/2011_09_30_drive_0018_sync',
        '2011_09_30/2011_09_30_drive_0027_sync'
    ]
    for odomseq in odomseqs:
        leftimgs = glob.glob(os.path.join(args.odom_root, odomseq, 'image_02/data', "*.png"))
        for leftimg in leftimgs:
            imgname = os.path.basename(leftimg)
            odomentries.append("{} {} {}".format(odomseq, imgname.rstrip('.png'), 'l'))
    return odomentries


def read_splits(args):
    split_root = os.path.join(project_rootdir, 'exp_pose_mdepth_kitti_eigen/splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'train_files.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'test_files.txt'), 'r')]
    odom_entries = get_odomentries(args)

    return odom_entries + train_entries + evaluation_entries

def remove_dup(entries):
    dupentry = list()
    for entry in entries:
        seq, index, _ = entry.split(' ')
        dupentry.append("{} {}".format(seq, index.zfill(10)))

    removed = list(set(dupentry))
    removed.sort()
    return removed

@torch.no_grad()
def validate_kitti_colorjitter(gpu, model, args, ngpus_per_node, eval_entries, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    stidx = 0
    edidx = len(eval_entries)
    print("Initialize Instance on Gpu %d, from %d to %d, total %d" % (gpu, stidx, edidx, len(eval_entries)))
    from tqdm import tqdm
    model.eval()
    model.cuda(gpu)

    totnum = 0
    dr = 0
    with torch.no_grad():
        for val_id, entry in enumerate(remove_dup(eval_entries[stidx : edidx])):
            seq, index = entry.split(' ')
            index = int(index)

            if os.path.exists(os.path.join(args.dataset, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))):
                tmproot = args.dataset
            else:
                tmproot = args.odom_root

            img1path = os.path.join(tmproot, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))
            img2path = os.path.join(tmproot, seq, 'image_02', 'data', "{}.png".format(str(index + 1).zfill(10)))

            if not os.path.exists(img2path):
                img2path = img1path

            image1 = frame_utils.read_gen(img1path)
            image2 = frame_utils.read_gen(img2path)

            image1 = np.array(image1).astype(np.uint8)
            image2 = np.array(image2).astype(np.uint8)

            image1 = torch.from_numpy(image1).permute([2, 0, 1]).float()
            image2 = torch.from_numpy(image2).permute([2, 0, 1]).float()

            image1 = image1[None].cuda(gpu)
            image2 = image2[None].cuda(gpu)

            padder = InputPadder(image1.shape, mode='kitti')
            image1, image2 = padder.pad(image1, image2)

            st = time.time()
            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            dr += time.time() - st
            totnum += 1
            print("%d Samples, Ave sec/frame: %f, Mem: %f Gb" % (totnum, dr / totnum, float(torch.cuda.memory_allocated() / 1024 / 1024 / 1024)))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--exportroot', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use small model')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--odom_root', type=str)

    parser.add_argument('--evalheight', type=int, default=1e10)
    parser.add_argument('--evalwidth', type=int, default=1e10)
    parser.add_argument('--maxinsnum', type=int, default=50)
    args = parser.parse_args()


    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    torch.manual_seed(1234)
    np.random.seed(1234)

    ngpus_per_node = torch.cuda.device_count()
    evaluation_entries = read_splits(args)
    validate_kitti_colorjitter(0, model.module, args, ngpus_per_node, evaluation_entries)