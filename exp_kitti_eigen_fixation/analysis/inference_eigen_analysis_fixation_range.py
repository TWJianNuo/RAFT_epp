from __future__ import print_function, division
import os, sys
project_rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

import argparse
import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from exp_kitti_eigen_fixation.analysis.dataset_kitti_eigen_analysis import KITTI_eigen

from mDnet import MDepthNet
import torch.utils.data as data
from PIL import Image, ImageDraw
from core.utils.flow_viz import flow_to_image
from core.utils.utils import InputPadder, forward_interpolate, tensor2disp, tensor2rgb, vls_ins, DistributedSamplerNoEvenlyDivisible
from posenet import Posenet
import torch.multiprocessing as mp
import torch.distributed as dist
import pickle
from torch.utils.data.sampler import Sampler
from detectron2.utils import comm
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def read_splits():
    split_root = os.path.join(project_rootdir, 'exp_pose_mdepth_kitti_eigen/splits')
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'test_files.txt'), 'r')]

    return evaluation_entries

class PoseMDNet(nn.Module):
    def __init__(self, args):
        super(PoseMDNet, self).__init__()

        import copy
        tmpargs = copy.deepcopy(args)
        tmpargs.inheight = 288
        tmpargs.inwidth = 960

        self.args = tmpargs
        self.deptmodel = MDepthNet(num_layers=self.args.num_layers, args=self.args)
        self.posemodel = Posenet(num_layers=self.args.num_layers, args=self.args)

    def forward(self, img1, img2):
        bz, _, h, w = img1.shape

        outputs = dict()
        outputs.update(self.deptmodel(img1))

        self_ang, self_tdir, self_tscale, obj_pose = self.posemodel(img1, img2)
        return outputs[('mDepth', 0)], self_tscale

def parse_input(image1, image2, h, w, args):
    inh = args.inheight
    inw = args.inwidth

    crps = [[0, inw, h-inh, h], [w-inw, w, h-inh, h]]
    image1s = list()
    image2s = list()
    for crp in crps:
        stx, edx, sty, edy = crp
        image1s.append(image1[:, :, sty:edy, stx:edx].clone())
        image2s.append(image2[:, :, sty:edy, stx:edx].clone())
    image1s = torch.cat(image1s, dim=0)
    image2s = torch.cat(image2s, dim=0)
    return image1s, image2s

class InferenceSampler(Sampler):
    """
    Produce indices for inference.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        shard_size = (self._size - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._size)
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

def train(gpu, ngpus_per_node, args):
    print("Using GPU %d for training" % gpu)
    args.gpu = gpu

    model = PoseMDNet(args=args)
    model = torch.nn.DataParallel(model)
    model.cuda()

    logroot = os.path.join(args.logroot, args.name)
    print("Parameter Count: %d, saving location: %s" % (count_parameters(model), logroot))

    if args.restore_ckpt is not None:
        print("=> loading checkpoint '{}'".format(args.restore_ckpt))
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.restore_ckpt, map_location=loc)
        model.load_state_dict(checkpoint, strict=False)

    model.train()

    evaluation_entries = read_splits()

    train_dataset = KITTI_eigen(root=args.dataset_root, inheight=args.inheight, inwidth=args.inwidth, entries=evaluation_entries,
                                ins_root=args.ins_root, depthvls_root=args.depthvlsgt_root, depth_root=args.depth_root, istrain=False, muteaug=True)
    dataloader = DataLoader(train_dataset, 1, num_workers=0, pin_memory=True, drop_last=False)
    model.eval()

    print("split length: %d, total length: %d" % (dataloader.__len__(), train_dataset.__len__()))

    diffrecs = list()
    with torch.no_grad():
        for i_batch, data_blob in enumerate(tqdm(dataloader)):
            image1 = data_blob['img1'].cuda(gpu) / 255.0
            image2 = data_blob['img2'].cuda(gpu) / 255.0
            backgorund = data_blob['insmap'].squeeze().numpy() == 0

            preddepth, predscale = model(image1, image2)

            preddepthnp = preddepth.squeeze().cpu().numpy()
            depthgtnp = data_blob['depthmap'].squeeze().cpu().numpy()

            gtscalenp = torch.sqrt(torch.sum(data_blob['rel_pose'][0, 0:3, 3] ** 2)).cpu().numpy()
            predscalenp = predscale.squeeze().cpu().numpy()

            selector = depthgtnp > 0 * backgorund
            diffrecs += (np.log(preddepthnp[selector] / predscalenp) - np.log(depthgtnp[selector] / gtscalenp)).tolist()

            # from core.utils.utils import tensor2disp
            # tensor2disp(1 / preddepth, vmax=0.15, viewind=0).show()

    diffrecs_abs = np.sort(np.abs(np.array(diffrecs)))
    num_sample = diffrecs_abs.shape[0]
    samplepose = np.linspace(0, int(num_sample * 0.999), 16).astype(np.int)

    sampled_edge = diffrecs_abs[samplepose]
    sampled_edge = np.sort(np.concatenate([sampled_edge[0:1], sampled_edge[1::], -sampled_edge[1::], np.array([-1.5])]))

    import matplotlib.pyplot as plt
    plt.hist(diffrecs, bins=100)
    for xc in sampled_edge:
        plt.axvline(x=xc, ymin=-0.1, ymax=0.1, color='r', linestyle='dashed', linewidth=0.8)
    plt.title('Log Difference with gt normlaized deoth')
    plt.savefig(os.path.join('/home/shengjie/Desktop/2021_03/2021_03_12', 'evaluation difference on log'))
    plt.close()

    import pickle
    with open('/home/shengjie/Documents/supporting_projects/RAFT/exp_kitti_eigen_fixation/eppflowenet/depth_bin.pickle', 'wb') as handle:
        pickle.dump(sampled_edge, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--inheight', type=int, default=288)
    parser.add_argument('--inwidth', type=int, default=1216)
    parser.add_argument('--evalheight', type=int, default=288)
    parser.add_argument('--evalwidth', type=int, default=1216)
    parser.add_argument('--maxinsnum', type=int, default=20)
    parser.add_argument('--min_depth_pred', type=float, default=1)
    parser.add_argument('--max_depth_pred', type=float, default=85)
    parser.add_argument('--min_depth_eval', type=float, default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, default=80)

    parser.add_argument('--tscale_range', type=float, default=3)
    parser.add_argument('--objtscale_range', type=float, default=10)
    parser.add_argument('--angx_range', type=float, default=0.03)
    parser.add_argument('--angy_range', type=float, default=0.06)
    parser.add_argument('--angz_range', type=float, default=0.01)
    parser.add_argument('--num_layers', type=int, default=50)

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--semantics_root', type=str)
    parser.add_argument('--depth_root', type=str)
    parser.add_argument('--depthvlsgt_root', type=str)
    parser.add_argument('--ins_root', type=str)
    parser.add_argument('--logroot', type=str)
    parser.add_argument('--num_workers', type=int, default=12)

    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dist_url', type=str, help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
    parser.add_argument('--dist_backend', type=str, help='distributed backend', default='nccl')

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir(os.path.join(args.logroot, args.name)):
        os.makedirs(os.path.join(args.logroot, args.name), exist_ok=True)
    os.makedirs(os.path.join(args.logroot, 'evaluation', args.name), exist_ok=True)

    torch.cuda.empty_cache()
    ngpus_per_node = torch.cuda.device_count()
    train(0, ngpus_per_node, args)