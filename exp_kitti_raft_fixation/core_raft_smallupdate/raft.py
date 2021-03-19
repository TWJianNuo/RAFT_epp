import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp_kitti_raft_fixation.core_raft_depth.update import BasicUpdateBlock, SmallUpdateBlock
from exp_kitti_raft_fixation.core_raft_depth.extractor import BasicEncoder, SmallEncoder
from exp_kitti_raft_fixation.core_raft_depth.corr import CorrBlock, AlternateCorrBlock
from core.utils.utils import bilinear_sampler, coords_grid, upflow8
import os
import pickle
from eppcore import eppcore_inflation, eppcore_compression

class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        samplesapce_log = np.linspace(0, args.sample_range * 2, args.num_deges - 1)
        sampled_rld = samplesapce_log - args.sample_range
        sampled_rld = np.sort(np.concatenate([np.array([0]), sampled_rld]))

        assert sampled_rld.shape[0] == args.num_deges
        self.nedges = args.num_deges
        self.sampled_rld = nn.Parameter(torch.from_numpy(sampled_rld).float().view(1, self.nedges, 1, 1), requires_grad=False)
        self.pts2ddict = dict()
        self.pts2ddict_org = dict()

        self.eppinflate = eppcore_inflation.apply
        self.eppcompress = eppcore_compression.apply

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_logdepth(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        init_logdepth = torch.log((torch.ones([N, 1, H//8, W//8]) * 10).to(img.device))
        return init_logdepth

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, 8*H, 8*W)

    def init_sampling_pts(self, bz, featureh, featurew, device, dsratio=8):
        infkey = "{}_{}_{}".format(bz, featureh, featurew)
        if infkey not in self.pts2ddict.keys():
            xx, yy = np.meshgrid(range(featurew), range(featureh), indexing='xy')
            xx = torch.from_numpy(xx).float().unsqueeze(0).unsqueeze(0).expand([bz, -1, -1, -1]).cuda(device)
            yy = torch.from_numpy(yy).float().unsqueeze(0).unsqueeze(0).expand([bz, -1, -1, -1]).cuda(device)
            ones = torch.ones_like(xx)
            self.pts2ddict[infkey] = (xx, yy, ones)

        h = int(featureh * dsratio)
        w = int(featurew * dsratio)
        infkey = "{}_{}_{}".format(bz, h, w)
        if infkey not in self.pts2ddict_org.keys():
            xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
            xx = torch.from_numpy(xx).float().unsqueeze(0).unsqueeze(0).expand([bz, -1, -1, -1]).cuda(device)
            yy = torch.from_numpy(yy).float().unsqueeze(0).unsqueeze(0).expand([bz, -1, -1, -1]).cuda(device)
            ones = torch.ones_like(xx)
            self.pts2ddict_org[infkey] = (xx, yy, ones)

    def forward(self, image1, image2, intrinsic, posepred, insmap, initialdepth=None, iters=12):
        """ Estimate optical flow between pair of frames """
        image2_org = torch.clone(image2)

        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        bz, _, featureh, featurew = fmap1.shape
        device = fmap1.device
        self.init_sampling_pts(bz, featureh, featurew, device)
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, intrinsic, posepred, insmap, nedges=self.nedges, maxinsnum=self.args.maxinsnum)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        bz, _, h, w = image1.shape
        if initialdepth is None:
            logdepth = self.initialize_logdepth(image1)
        else:
            logdepth = F.interpolate(torch.log(torch.clamp_min(initialdepth, min=1)), [int(h / 8), int(w / 8)], mode='nearest')

        depth_predictions = []
        logdepth_predictions = []
        local_sample_pts2ds = []
        delta_depths = []
        outputs = dict()
        for itr in range(iters):
            corr, local_sample_pts2d = corr_fn(logdepth, sampled_rld=self.sampled_rld, pts2ddict=self.pts2ddict) # index correlation volume

            net, up_mask, delta_depth = self.update_block(net, inp, corr, logdepth)

            # F(t+1) = F(t) + \Delta(t)
            logdepth = logdepth + delta_depth
            depth = torch.exp(logdepth)

            depth_up = self.upsample_flow(depth, up_mask)

            if itr == iters - 1:
                flowpred, img1_recon, projMimg = self.depth2rgb(depth_up, image2_org, intrinsic, posepred, insmap)
                outputs['flowpred'] = flowpred
                outputs['img1_recon'] = img1_recon
                # with torch.no_grad():
                #     outputs['orgflow'] = self.depth2flow(initialdepth, projMimg)
            
            depth_predictions.append(depth_up)
            logdepth_predictions.append(F.interpolate(logdepth, [h, w], mode='bilinear', align_corners=False))
            local_sample_pts2ds.append(local_sample_pts2d)
            delta_depths.append(delta_depth)

        outputs['depth_predictions'] = depth_predictions
        outputs['logdepth_predictions'] = logdepth_predictions
        outputs['local_sample_pts2ds'] = local_sample_pts2ds
        outputs['delta_depths'] = delta_depths
        return outputs

    def depth2rgb(self, depthmap, img2, intrinsic, posepred, insmap):
        bz, _, orgh, orgw = depthmap.shape
        infkey = "{}_{}_{}".format(bz, orgh, orgw)
        xx, yy, ones = self.pts2ddict_org[infkey]

        intrinsicex = intrinsic.unsqueeze(1).expand([-1, self.args.maxinsnum, -1, -1])
        projM = intrinsicex @ posepred @ torch.inverse(intrinsicex)
        projMimg = self.eppinflate(insmap, projM)

        pts3d = torch.stack([xx * depthmap, yy * depthmap, depthmap, ones], dim=-1).squeeze(1).unsqueeze(-1)
        pts2dp = projMimg @ pts3d

        pxx, pyy, pzz, _ = torch.split(pts2dp, 1, dim=3)

        sign = pzz.sign()
        sign[sign == 0] = 1
        pzz = torch.clamp(torch.abs(pzz), min=1e-20) * sign

        pxx = (pxx / pzz).squeeze(-1).squeeze(-1)
        pyy = (pyy / pzz).squeeze(-1).squeeze(-1)
        pzz = pzz.squeeze(-1).squeeze(-1)

        flowpredx = pxx - xx.squeeze(1)
        flowpredy = pyy - yy.squeeze(1)
        flowpred = torch.stack([flowpredx, flowpredy], dim=1)

        supressval = torch.ones_like(pxx) * (-100)
        inboundmask = (pzz > 1e-20).float()

        pxx = inboundmask * pxx + supressval * (1 - inboundmask)
        pyy = inboundmask * pyy + supressval * (1 - inboundmask)

        pxx = (pxx / orgw - 0.5) * 2
        pyy = (pyy / orgh - 0.5) * 2
        samplepts = torch.stack([pxx, pyy], dim=-1)
        img1_recon = F.grid_sample(img2, samplepts, mode='bilinear', align_corners=False)
        return flowpred, img1_recon, projMimg

    def depth2flow(self, depthmap, projMimg):
        bz, _, orgh, orgw = depthmap.shape
        infkey = "{}_{}_{}".format(bz, orgh, orgw)
        xx, yy, ones = self.pts2ddict_org[infkey]

        pts3d = torch.stack([xx * depthmap, yy * depthmap, depthmap, ones], dim=-1).squeeze(1).unsqueeze(-1)
        pts2dp = projMimg @ pts3d

        pxx, pyy, pzz, _ = torch.split(pts2dp, 1, dim=3)

        sign = pzz.sign()
        sign[sign == 0] = 1
        pzz = torch.clamp(torch.abs(pzz), min=1e-20) * sign

        pxx = (pxx / pzz).squeeze(-1).squeeze(-1)
        pyy = (pyy / pzz).squeeze(-1).squeeze(-1)

        flowpredx = pxx - xx.squeeze(1)
        flowpredy = pyy - yy.squeeze(1)
        flowpred = torch.stack([flowpredx, flowpredy], dim=1)
        return flowpred