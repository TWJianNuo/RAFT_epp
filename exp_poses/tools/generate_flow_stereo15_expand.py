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
from exp_poses.dataset_kitti_stereo15_orged import KITTI_eigen_stereo15

import core.datasets as datasets
from core.utils import flow_viz, frame_utils
from core.raft import RAFT
from core.utils.utils import InputPadder, forward_interpolate
from numpy.random import default_rng
import matplotlib.pyplot as plt

def readlines(filename):
    with open(filename, 'r') as f:
        filenames = f.readlines()
    return filenames

def read_splits_mapping(args):
    evaluation_entries = []
    import glob
    for m in range(200):
        seqname = "kittistereo15_{}/kittistereo15_{}_sync".format(str(m).zfill(6), str(m).zfill(6))
        evaluation_entries.append("{} {} {}".format(seqname, "10".zfill(10), 'l'))

    expandentries = list()
    mappings = readlines(args.mpf_root)
    for idx, m in enumerate(mappings):
        if len(m) == 1:
            continue
        d, s, cidx = m.split(' ')
        seq = "{}/{}".format(d, s)
        pngs = glob.glob(os.path.join(args.dataset, d, s, 'image_02/data', '*.png'))
        for p in pngs:
            frmidx = p.split('/')[-1].split('.')[0]
            expandentries.append("{} {} l".format(seq, frmidx.zfill(10)))
    expandentries = list(set(expandentries))
    expandentries.sort()
    return expandentries

@torch.no_grad()
def validate_kitti_colorjitter(model, args, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    from tqdm import tqdm
    model.eval()

    expandentries = read_splits_mapping(args)

    for val_id, entry in enumerate(tqdm(expandentries)):
        seq, idx, _ = entry.split(' ')
        idx = int(idx)
        img1path = os.path.join(args.dataset, seq, 'image_02', 'data', "{}.png".format(str(idx).zfill(10)))
        img2path = os.path.join(args.dataset, seq, 'image_02', 'data', "{}.png".format(str(idx + 1).zfill(10)))

        if not os.path.exists(img2path):
            img2path = img1path

        img1 = np.array(Image.open(img1path)).astype(np.float32)
        img2 = np.array(Image.open(img2path)).astype(np.float32)

        img1 = torch.from_numpy(img1).permute([2, 0, 1]).unsqueeze(0)
        img2 = torch.from_numpy(img2).permute([2, 0, 1]).unsqueeze(0)

        svfold = os.path.join(args.exportroot, seq, 'image_02')
        svpath = os.path.join(args.exportroot, seq, 'image_02', "{}.png".format(str(idx).zfill(10)))
        os.makedirs(svfold, exist_ok=True)

        image1 = img1.cuda()
        image2 = img2.cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        frame_utils.writeFlowKITTI(svpath, flow.permute(1, 2, 0).numpy())
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--exportroot', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--mpf_root', type=str)

    parser.add_argument('--evalheight', type=int, default=1e10)
    parser.add_argument('--evalwidth', type=int, default=1e10)
    parser.add_argument('--maxinsnum', type=int, default=50)
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    torch.manual_seed(1234)
    np.random.seed(1234)
    with torch.no_grad():
        validate_kitti_colorjitter(model.module, args)