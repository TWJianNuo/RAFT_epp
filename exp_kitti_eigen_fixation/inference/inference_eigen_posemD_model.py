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
from exp_kitti_eigen_fixation.inference.dataset_kitti_eigen_inference import KITTI_eigen

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
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

    if args.only_eval:
        return evaluation_entries
    else:
        if args.ban_odometry:
            return train_entries + evaluation_entries
        else:
            return train_entries + evaluation_entries + odom_entries

class PoseMDNet(nn.Module):
    def __init__(self, args):
        super(PoseMDNet, self).__init__()
        self.args = args
        self.deptmodel = MDepthNet(num_layers=args.num_layers, args=args)
        self.posemodel = Posenet(num_layers=args.num_layers, args=args)
        self.pts2ddict = dict()

    def combine_back(self, orgh, orgw, outputs):
        inw = self.args.inwidth
        inh = self.args.inheight
        crps = [[0, inw, orgh - inh, orgh], [orgw - inw, orgw, orgh - inh, orgh]]

        device = outputs[('mDepth', 0)].device
        predcounts = torch.zeros([1, 1, orgh, orgw], device=device)
        preddepth = torch.zeros([1, 1, orgh, orgw], device=device)
        objscale = torch.zeros([1, 1, orgh, orgw], device=device)
        objang = torch.zeros([1, 1, orgh, orgw], device=device)
        for idx, crp in enumerate(crps):
            stx, edx, sty, edy = crp
            predcounts[:, :, sty:edy, stx:edx] += 1
            preddepth[:, :, sty:edy, stx:edx] += outputs[('mDepth', 0)][idx].unsqueeze(0)
            objscale[:, :, sty:edy, stx:edx] += outputs[('obj_scale', 0)][idx].unsqueeze(0)
            objang[:, :, sty:edy, stx:edx] += outputs[('obj_angle', 0)][idx].unsqueeze(0)

        preddepth = preddepth / (predcounts + 1e-10)
        objscale = objscale / (predcounts + 1e-10)
        objang = objang / (predcounts + 1e-10)
        selfRT = torch.mean(outputs['selfpose'], dim=0, keepdim=True)
        # tensor2disp(1 / preddepth, vmax=0.15, viewind=0).show()
        return preddepth, objscale, objang, selfRT

    def forward(self, img1, img2, insmap, orgh, orgw):
        bz, _, h, w = img1.shape

        outputs = dict()
        outputs.update(self.deptmodel(img1))

        self_ang, self_tdir, self_tscale, obj_pose = self.posemodel(img1, img2)
        selfRT = self.posemodel.get_selfpose(selfang=self_ang, selftdir=self_tdir, selfscale=self_tscale)

        outputs['selfpose'] = selfRT
        outputs.update(obj_pose)
        preddepth, objscale, objang, selfRT = self.combine_back(orgh, orgw, outputs)

        maxinsnum = insmap.max().item() + 1
        insnum = self.posemodel.eppcompress(insmap, (insmap > -1).float().squeeze(1).unsqueeze(-1).unsqueeze(-1), maxinsnum)

        obj_scalep_cps = self.posemodel.eppcompress(insmap, objscale.squeeze(1).unsqueeze(-1).unsqueeze(-1), maxinsnum)
        obj_scalep_cps = obj_scalep_cps / (insnum + 1e-10)

        obj_angp_cps = self.posemodel.eppcompress(insmap, objang.squeeze(1).unsqueeze(-1).unsqueeze(-1), maxinsnum)
        obj_angp_cps = obj_angp_cps / (insnum + 1e-10)

        predposes = self.posemodel.mvinfo2objpose(obj_angp_cps, obj_scalep_cps, selfRT)
        return preddepth, predposes

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

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

def train(gpu, ngpus_per_node, args):
    print("Using GPU %d for training" % gpu)
    args.gpu = gpu

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=ngpus_per_node, rank=args.gpu)

    model = PoseMDNet(args=args)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(module=model)
        model = model.to(f'cuda:{args.gpu}')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True, output_device=args.gpu)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    if args.restore_ckpt is not None:
        print("=> loading checkpoint '{}'".format(args.restore_ckpt))
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.restore_ckpt, map_location=loc)
        model.load_state_dict(checkpoint, strict=False)

    model.train()

    inference_entries = read_splits(args)

    train_dataset = KITTI_eigen(root=args.dataset_root, odom_root=args.odom_root, inheight=args.inheight, inwidth=args.inwidth, entries=inference_entries, ins_root=args.ins_root)
    sampler = DistributedSamplerNoEvenlyDivisible(train_dataset, shuffle=False)
    dataloader = DataLoader(train_dataset, 1, num_workers=4, pin_memory=True, sampler=sampler)
    model.eval()

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    print("current data sampler rank is %d, replicas is %d, split length: %d, total length: %d, sampledataset entry: %s" % (rank, sampler.__len__(), train_dataset.__len__(), world_size, train_dataset.entries[300]))

    with torch.no_grad():
        for i_batch, data_blob in enumerate(tqdm(dataloader)):
            image1 = data_blob['img1'].cuda(gpu) / 255.0
            image2 = data_blob['img2'].cuda(gpu) / 255.0
            insmap = data_blob['insmap'].cuda(gpu)
            tag = data_blob['tag'][0]

            seq, frmidx = tag.split(' ')
            exportfold_depth = os.path.join(args.export_root, seq, 'image_02/depthpred')
            exportfold_pose = os.path.join(args.export_root, seq, 'image_02/posepred')

            os.makedirs(exportfold_depth, exist_ok=True)
            os.makedirs(exportfold_pose, exist_ok=True)

            exportpath_depth = os.path.join(args.export_root, seq, 'image_02/depthpred', "{}.png".format(str(frmidx).zfill(10)))
            exportpath_pose = os.path.join(args.export_root, seq, 'image_02/posepred', "{}.pickle".format(str(frmidx).zfill(10)))

            if os.path.isfile(exportpath_depth) and os.path.isfile(exportpath_pose):
                print("%s exists" % tag)
                raise Exception("Duplicate!")

            _, _, orgh, orgw = image1.shape

            image1s, image2s = parse_input(image1, image2, orgh, orgw, args)

            preddepth, predposes = model(image1s, image2s, insmap, orgh, orgw)

            preddepthnp = preddepth[0].squeeze().cpu().numpy() * 256.0
            preddepthnp = np.array(preddepthnp).astype(np.uint16)
            Image.fromarray(preddepthnp).save(exportpath_depth)

            predposesnp = predposes[0].cpu().numpy()
            with open(exportpath_pose, 'wb') as handle:
                pickle.dump(predposesnp, handle)
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
    parser.add_argument('--inwidth', type=int, default=960)
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
    parser.add_argument('--odom_root', type=str)
    parser.add_argument('--export_root', type=str)
    parser.add_argument('--ins_root', type=str)
    parser.add_argument('--logroot', type=str)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--only_eval', action='store_true')
    parser.add_argument('--ban_odometry', action='store_true')

    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dist_url', type=str, help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
    parser.add_argument('--dist_backend', type=str, help='distributed backend', default='nccl')

    args = parser.parse_args()
    args.dist_url = args.dist_url.rstrip('1234') + str(np.random.randint(2000, 3000, 1).item())

    torch.manual_seed(1234)
    np.random.seed(1234)

    torch.cuda.empty_cache()

    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        args.world_size = ngpus_per_node
        mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        train(args.gpu, ngpus_per_node, args)