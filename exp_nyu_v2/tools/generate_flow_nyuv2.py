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


def read_splits():
    split_root = os.path.join(project_rootdir, 'exp_nyu_v2/splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'nyudepthv2_train_files.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'nyudepthv2_test_files.txt'), 'r')]
    return train_entries + evaluation_entries

def remove_dup(entries):
    removed = list(set(entries))
    removed.sort()
    return removed

@torch.no_grad()
def validate_kitti_colorjitter(gpu, model, args, ngpus_per_node, eval_entries, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    interval = np.floor(len(eval_entries) / ngpus_per_node).astype(np.int).item()
    if gpu == ngpus_per_node - 1:
        stidx = int(interval * gpu)
        edidx = len(eval_entries)
    else:
        stidx = int(interval * gpu)
        edidx = int(interval * (gpu + 1))
    print("Initialize Instance on Gpu %d, from %d to %d, total %d" % (gpu, stidx, edidx, len(eval_entries)))
    from tqdm import tqdm
    model.eval()
    model.cuda(gpu)
    with torch.no_grad():
        for val_id, entry in enumerate(tqdm(remove_dup(eval_entries[stidx : edidx]))):
            seq, index = entry.split(' ')
            index = int(index)

            img1path = os.path.join(args.dataset, seq, 'rgb_{}.jpg'.format(str(index).zfill(5)))
            img2path = os.path.join(args.dataset, seq, 'rgb_{}.jpg'.format(str(index + 1).zfill(5)))

            if not os.path.exists(img2path):
                img2path = img1path

            image1 = frame_utils.read_gen(img1path)
            image2 = frame_utils.read_gen(img2path)

            image1 = np.array(image1).astype(np.uint8)
            image2 = np.array(image2).astype(np.uint8)

            image1 = torch.from_numpy(image1).permute([2, 0, 1]).float()
            image2 = torch.from_numpy(image2).permute([2, 0, 1]).float()

            svfold = os.path.join(args.exportroot, seq, )
            svpath = os.path.join(args.exportroot, seq, '{}.png'.format(str(index).zfill(5)))
            os.makedirs(svfold, exist_ok=True)

            image1 = image1[None].cuda(gpu)
            image2 = image2[None].cuda(gpu)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = flow_pr.squeeze(0)
            flow_numpy = flow.cpu().permute(1, 2, 0).numpy()

            frame_utils.writeFlowKITTI(svpath, flow_numpy)

            if args.dovls:
                vlsflow = Image.fromarray(flow_viz.flow_to_image(flow_numpy))
                vlsmrgb1 = tensor2rgb(image1 / 255.0, viewind=0)
                vlsrgb2 = tensor2rgb(image2 / 255.0, viewind=0)

                w, h = vlsrgb2.size
                xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
                xxf = xx.flatten()
                yyf = yy.flatten()
                rndidx = np.random.randint(0, xxf.shape[0], 100)

                xxf_rnd = xxf[rndidx]
                yyf_rnd = yyf[rndidx]

                flowx = flow_numpy[yyf_rnd, xxf_rnd, 0]
                flowy = flow_numpy[yyf_rnd, xxf_rnd, 1]

                fig = plt.figure(figsize=(12, 9))
                plt.subplot(2, 1, 1)
                plt.scatter(xxf_rnd, yyf_rnd,  1, 'r')
                plt.imshow(vlsrgb1)
                plt.subplot(2, 1, 2)
                plt.scatter(flowx + xxf_rnd, flowy + yyf_rnd,  1, 'r')
                plt.imshow(vlsrgb2)
                plt.savefig(os.path.join("/media/shengjie/disk1/Prediction/nyuv2_flow_vls", "{}_{}.jpg".format(seq, str(index).zfill(5))))
                plt.close()

                # combined_left = np.concatenate([np.array(vlsrgb1), np.array(vlsrgb2)], axis=0)
                # combined_right = np.concatenate([np.array(vlsflow), np.array(vlsflow)], axis=0)
                # combined = np.concatenate([combined_left, combined_right], axis=1)
                # vls_sv_root = os.makedirs("/media/shengjie/disk1/Prediction/nyuv2_flow_vls", exist_ok=True)
                # Image.fromarray(combined).save(os.path.join("/media/shengjie/disk1/Prediction/nyuv2_flow_vls", "{}_{}.jpg".format(seq, str(index).zfill(5))))

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
    parser.add_argument('--dovls', action='store_true', help='use small model')

    parser.add_argument('--evalheight', type=int, default=1e10)
    parser.add_argument('--evalwidth', type=int, default=1e10)
    parser.add_argument('--maxinsnum', type=int, default=50)
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    torch.manual_seed(1234)
    np.random.seed(1234)

    ngpus_per_node = torch.cuda.device_count()
    evaluation_entries = read_splits()
    evaluation_entries.sort()

    mp.spawn(validate_kitti_colorjitter, nprocs=ngpus_per_node, args=(model.module, args, ngpus_per_node, evaluation_entries))