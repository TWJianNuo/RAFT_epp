import torch
import torch.nn.functional as F
import numpy as np
from core.utils.utils import bilinear_sampler, coords_grid
from eppcore import eppcore_inflation, eppcore_compression

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, intrinsic, posepred, insmap, nedges, maxinsnum, num_levels=4, dsratio=8):
        self.num_levels = num_levels
        self.dsratio = dsratio
        self.corr_pyramid = []
        self.eppinflate = eppcore_inflation.apply
        self.eppcompress = eppcore_compression.apply
        self.nedges = nedges
        self.pts2dsampledict = dict()

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        self.corr = corr.reshape(batch*h1*w1, dim, h2, w2)

        # Compute Sampling Positions
        bz, _, featureh, featurew = fmap1.shape
        orgh = int(featureh * dsratio)
        orgw = int(featurew * dsratio)

        self.featureh = featureh
        self.featurew = featurew
        self.orgh = orgh
        self.orgw = orgw
        self.maxinsnum = maxinsnum
        self.bz = bz

        resizeM = torch.eye(4)
        resizeM[0, 0] = 1 / dsratio
        resizeM[1, 1] = 1 / dsratio
        resizeM = resizeM.view([1, 4, 4]).expand([self.bz, -1, -1])
        resizeM = resizeM.cuda(intrinsic.device)

        intrinsic_resized = resizeM @ intrinsic

        intrinsicex = intrinsic_resized.unsqueeze(1).expand([-1, self.maxinsnum, -1, -1])
        projM = intrinsicex @ posepred @ torch.inverse(intrinsicex)
        projMimg = self.eppinflate(insmap, projM)
        projMimg = F.interpolate(projMimg.permute([0, 3, 4, 1, 2]).view([self.bz, 16, self.orgh, self.orgw]), [self.featureh, self.featurew], mode='nearest')
        self.projMimg = projMimg.permute([0, 2, 3, 1]).view([self.bz, self.featureh, self.featurew, 4, 4])

    def __call__(self, logdepth, sampled_rld, pts2ddict):
        sample_rld = sampled_rld.expand([self.bz, -1, self.featureh, self.featurew])
        sample_ld = logdepth.expand([-1, self.nedges, -1, -1]) + sample_rld

        sample_d = torch.exp(sample_ld)

        infkey = "{}_{}_{}".format(self.bz, self.featureh, self.featurew)
        xx, yy, ones = pts2ddict[infkey]

        xx = xx.expand([-1, self.nedges, -1, -1])
        yy = yy.expand([-1, self.nedges, -1, -1])
        ones = ones.expand([-1, self.nedges, -1, -1])

        pts3d = torch.stack([xx * sample_d, yy * sample_d, sample_d, ones], dim=-1).unsqueeze(-1)
        pts2dp = self.projMimg.unsqueeze(1).expand([-1, self.nedges, -1, -1, -1, -1]) @ pts3d

        pxx, pyy, pzz, _ = torch.split(pts2dp, 1, dim=4)

        sign = pzz.sign()
        sign[sign == 0] = 1
        pzz = torch.clamp(torch.abs(pzz), min=1e-20) * sign

        pxx = (pxx / pzz).squeeze(-1).squeeze(-1)
        pyy = (pyy / pzz).squeeze(-1).squeeze(-1)

        supressval = torch.ones_like(pxx) * (-100)
        inboundmask = (pzz > 1e-20).squeeze(-1).squeeze(-1).float()

        pxx = inboundmask * pxx + supressval * (1 - inboundmask)
        pyy = inboundmask * pyy + supressval * (1 - inboundmask)

        sample_px = (pxx / self.featurew - 0.5) * 2
        sample_py = (pyy / self.featureh - 0.5) * 2

        sample_px = sample_px.permute([0, 2, 3, 1]).contiguous().view(self.bz * self.featureh * self.featurew, 1, self.nedges)
        sample_py = sample_py.permute([0, 2, 3, 1]).contiguous().view(self.bz * self.featureh * self.featurew, 1, self.nedges)
        local_sample_pts2d = torch.stack([sample_px, sample_py], dim=-1)

        out = bilinear_sampler(self.corr, local_sample_pts2d)
        out = out.view(self.bz, self.featureh, self.featurew, -1)

        return out.permute(0, 3, 1, 2).contiguous().float(), local_sample_pts2d.view([self.bz, self.featureh, self.featurew, self.nedges, 2])

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
