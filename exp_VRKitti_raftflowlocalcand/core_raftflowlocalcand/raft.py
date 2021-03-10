import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp_VRKitti_raftflowlocalcand.core_raftflowlocalcand.update import BasicUpdateBlock, SmallUpdateBlock
from exp_VRKitti_raftflowlocalcand.core_raftflowlocalcand.extractor import BasicEncoder, SmallEncoder
from exp_VRKitti_raftflowlocalcand.core_raftflowlocalcand.corr import CorrBlock
from core.utils.utils import bilinear_sampler, coords_grid, upflow8

from exp_VRKitti_raftflowlocalcand.core_raftflowlocalcand.depthnet import MDepthNet

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, flow_init, iters=12, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        coords1 = coords1 + flow_init
        mag = torch.sqrt(torch.sum(flow_init ** 2, dim=1, keepdim=True) + 1e-10)

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()

            corr, coords_lvl, delta_lvl = corr_fn(coords1, torch.ones_like(mag), self.args.samplewindowr) # index correlation volume

            if itr == 0:
                coords_lvl_vls = coords_lvl

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow, prob = self.update_block(net, inp, corr, flow, delta_lvl)

            if itr == 0:
                prob_vls = prob

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions, coords_lvl_vls, prob_vls

class MDiRaft(nn.Module):
    def __init__(self, args):
        super(MDiRaft, self).__init__()
        self.args = args
        self.raft = RAFT(self.args)
        self.mDd = MDepthNet(num_layers=50, args=self.args)

    def depth2flow(self, depthmap, pMImg):
        bz, _, h, w = depthmap.shape

        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        ones = np.ones_like(xx)
        xx = torch.from_numpy(xx).float().unsqueeze(0).unsqueeze(0).expand([bz, -1, -1, -1]).cuda(depthmap.device)
        yy = torch.from_numpy(yy).float().unsqueeze(0).unsqueeze(0).expand([bz, -1, -1, -1]).cuda(depthmap.device)
        ones = torch.from_numpy(ones).float().unsqueeze(0).unsqueeze(0).expand([bz, -1, -1, -1]).cuda(depthmap.device)

        pts3d = torch.stack([xx * depthmap, yy * depthmap, depthmap, ones], dim=-1).unsqueeze(-1)
        pts2d = pMImg @ pts3d

        pts2dx, pts2dy, pts2dz, ones = torch.split(pts2d, 1, dim=4)

        sign = pts2dz.sign()
        sign[sign == 0] = 1
        pts2dz = torch.clamp(torch.abs(pts2dz), min=1e-20) * sign

        flowx = (pts2dx / pts2dz).squeeze(-1).squeeze(-1) - xx
        flowy = (pts2dy / pts2dz).squeeze(-1).squeeze(-1) - yy
        flowpred = torch.cat([flowx, flowy], dim=1)
        return flowpred

    def forward(self, image1, image2, pMImg, iters=12, test_mode=False, depthmap=None):
        dsratio = 8.0
        bz, _, h, w = image1.shape
        mDdoutputs = self.mDd(image1)
        if depthmap is None:
            initflow = self.depth2flow(mDdoutputs[('mDepth', 0)], pMImg).detach()
        else:
            initflow = self.depth2flow(depthmap, pMImg).detach()
        initflow_ds = F.interpolate(initflow, [int(h / dsratio), int(w / dsratio)], mode='nearest') / dsratio

        # check code for downsample
        # cky = np.random.randint(0, int(h/dsratio), 1).item()
        # ckx = np.random.randint(0, int(w/dsratio), 1).item()
        # dsval = initflow_ds[0, :, cky, ckx] * dsratio
        # ckval = initflow[0, :, int(cky * dsratio), int(ckx * dsratio)]

        if not test_mode:
            flow_predictions, coords_lvl_vls, prob_vls = self.raft(image1, image2, flow_init=initflow_ds, iters=iters, test_mode=test_mode)
            return mDdoutputs, initflow, flow_predictions, coords_lvl_vls, prob_vls
        else:
            flow_low, flow_up = self.raft(image1, image2, flow_init=initflow_ds, iters=iters, test_mode=test_mode)
            return mDdoutputs, flow_low, flow_up
