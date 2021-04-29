import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from collections import OrderedDict
from eppcore import eppcore_inflation, eppcore_compression

class Hourglass2D_down(nn.Module):
    def __init__(self, n, dimin, dimout, expand=64):
        super(Hourglass2D_down, self).__init__()
        self.n = n
        self.expand = expand

        self.conv1 = Conv2d(dimin=dimin, dimout=dimout)
        self.conv2 = Conv2d(dimin=dimout, dimout=dimout)

        self.padding = nn.ReplicationPad2d((1, 0, 1, 0))
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = Conv2d(dimin=dimout, dimout=dimout+self.expand)

        if self.n <= 1:
            self.endconv = Conv2d(dimin=dimout+self.expand, dimout=dimout+self.expand)

    def forward(self, x):
        x = x + self.conv2(self.conv1(x))
        y = self.pool1(self.padding(x))
        y = self.conv3(y)
        if self.n == 1:
            y = self.endconv(y)
        elif self.n < 1:
            raise Exception("Invalid Index")

        return y, x

class Hourglass2D_up(nn.Module):
    def __init__(self, dimin, dimout):
        super(Hourglass2D_up, self).__init__()
        self.conv1 = Conv2d(dimin=dimin, dimout=dimout)

    def forward(self, x, residual):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = x + residual
        return x

class Hourglass2D(nn.Module):
    def __init__(self, n, dimin, expand=64):
        super(Hourglass2D, self).__init__()
        self.n = n
        for i in range(self.n, 0, -1):
            cdimin = dimin + (self.n - i) * expand
            cdimout = cdimin
            self.__setattr__('hdowns_{}'.format(i), Hourglass2D_down(i, cdimin, cdimout, expand=expand))
            self.__setattr__('hups_{}'.format(i), Hourglass2D_up(cdimin + expand, cdimin))

    def forward(self, x):
        residuals = OrderedDict()
        for i in range(self.n, 0, -1):
            x, residual = self.__getattr__('hdowns_{}'.format(i))(x)
            residuals['residual_{}'.format(i)] = residual

        for i in range(1, self.n + 1):
            x = self.__getattr__('hups_{}'.format(i))(x, residuals['residual_{}'.format(i)])

        return x

class Hourglass3D_down(nn.Module):
    def __init__(self, n, dimin, dimout, expand=64):
        super(Hourglass3D_down, self).__init__()
        self.n = n
        self.expand = expand

        self.conv1 = Conv3d(dimin=dimin, dimout=dimout)
        self.conv2 = Conv3d(dimin=dimout, dimout=dimout)

        self.padding = nn.ReplicationPad3d((1, 0, 1, 0, 1, 0))
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = Conv3d(dimin=dimout, dimout=dimout+self.expand)

        if self.n <= 1:
            self.endconv = Conv3d(dimin=dimout+self.expand, dimout=dimout+self.expand)

    def forward(self, x):
        x = x + self.conv2(self.conv1(x))
        y = self.pool1(self.padding(x))
        y = self.conv3(y)
        if self.n == 1:
            y = self.endconv(y)
        elif self.n < 1:
            raise Exception("Invalid Index")

        return y, x

class Hourglass3D_up(nn.Module):
    def __init__(self, dimin, dimout):
        super(Hourglass3D_up, self).__init__()
        self.conv1 = Conv3d(dimin=dimin, dimout=dimout)

    def forward(self, x, residual):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = x + residual
        return x

class Hourglass3D(nn.Module):
    def __init__(self, n, dimin, expand=64):
        super(Hourglass3D, self).__init__()
        self.n = n
        for i in range(self.n, 0, -1):
            cdimin = dimin + (self.n - i) * expand
            cdimout = cdimin
            self.__setattr__('hdowns_{}'.format(i), Hourglass3D_down(i, cdimin, cdimout, expand=expand))
            self.__setattr__('hups_{}'.format(i), Hourglass3D_up(cdimin + expand, cdimin))

    def forward(self, x):
        residuals = OrderedDict()
        for i in range(self.n, 0, -1):
            x, residual = self.__getattr__('hdowns_{}'.format(i))(x)
            residuals['residual_{}'.format(i)] = residual

        for i in range(1, self.n + 1):
            x = self.__getattr__('hups_{}'.format(i))(x, residuals['residual_{}'.format(i)])

        return x

class BnRelu(nn.Module):
    def __init__(self, dimin):
        super(BnRelu, self).__init__()
        self.norm2d = nn.BatchNorm2d(num_features=dimin, momentum=0.05, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm2d(x))

class Conv2d(nn.Module):
    def __init__(self, dimin, dimout, stride=1, bn=True):
        super(Conv2d, self).__init__()
        if bn:
            self.bnrelu = BnRelu(dimin)
        else:
            self.bnrelu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=dimin, out_channels=dimout, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        return self.conv(self.bnrelu(x))

class BnRelu3d(nn.Module):
    def __init__(self, dimin):
        super(BnRelu3d, self).__init__()
        self.norm3d = nn.BatchNorm3d(num_features=dimin, momentum=0.05, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm3d(x))

class Conv3d(nn.Module):
    def __init__(self, dimin, dimout, stride=1, bn=True):
        super(Conv3d, self).__init__()
        if bn:
            self.bnrelu = BnRelu3d(dimin)
        else:
            self.bnrelu = nn.ReLU()
        self.conv = nn.Conv3d(in_channels=dimin, out_channels=dimout, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        return self.conv(self.bnrelu(x))

class ResConv2d(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, dimin, dimout, stride=1):
        super(ResConv2d, self).__init__()
        self.stride = stride
        if self.stride == 1:
            self.conv1 = Conv2d(dimin, dimout, stride=1)
            self.conv2 = Conv2d(dimout, dimout, stride=1)
        elif self.stride == 2:
            self.conv1 = Conv2d(dimin, dimout, stride=1)
            self.conv2 = Conv2d(dimout, dimout, stride=2)
            self.conv3 = nn.Conv2d(in_channels=dimin, out_channels=dimout, kernel_size=1, stride=2)
        else:
            raise Exception("Stride not support")

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.stride == 2:
            x = self.conv3(x)
        return x + y

class EppFlowNet_Encoder(nn.Module):
    def __init__(self):
        super(EppFlowNet_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=3, stride=2)
        self.resconv1 = ResConv2d(dimin=32, dimout=32, stride=1)
        self.resconv2 = ResConv2d(dimin=32, dimout=32, stride=1)
        self.resconv3 = ResConv2d(dimin=32, dimout=32, stride=1)

        self.resconv4 = ResConv2d(dimin=32, dimout=64, stride=2)

        self.resconv5 = ResConv2d(dimin=64, dimout=64, stride=1)
        self.resconv6 = ResConv2d(dimin=64, dimout=64, stride=1)
        self.resconv7 = ResConv2d(dimin=64, dimout=64, stride=1)

        self.hug2d1 = Hourglass2D(n=4, dimin=64)
        self.hug2d2 = Hourglass2D(n=4, dimin=64)

        self.convf = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)

    def forward(self, image):
        x = self.conv1(image)
        x = self.resconv1(x)
        x = self.resconv2(x)
        x = self.resconv3(x)

        x = self.resconv4(x)

        x = self.resconv5(x)
        x = self.resconv6(x)
        x = self.resconv7(x)

        x = self.hug2d1(x)
        x = self.hug2d2(x)

        x = self.convf(x)
        return x

class Scalehead(nn.Module):
    def __init__(self, args):
        super(Scalehead, self).__init__()
        self.bnrelu = BnRelu3d(48)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(in_channels=48, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=int(32 / args.num_angs), out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.args = args

    def forward(self, x):
        _, _, _, featureh, featurew = x.shape
        x = self.bnrelu(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x).squeeze(1))

        xs = torch.split(x, dim=1, split_size_or_sections=int(32 / self.args.num_angs))
        xxs = list()
        for xx in xs:
            xx = self.conv4(xx)
            xxs.append(xx)
        xxs = torch.cat(xxs, dim=1)
        pred = F.interpolate(xxs, [int(featureh * 4), int(featurew * 4)], mode='bilinear', align_corners=False)
        pred = (self.sigmoid(pred) - 0.5) * 2 * self.args.maxlogscale
        return pred

class Relposehead(nn.Module):
    def __init__(self, args):
        super(Relposehead, self).__init__()
        self.bnrelu = BnRelu3d(48)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(in_channels=48, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=args.num_angs, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.args = args

    def forward(self, x):
        _, _, _, featureh, featurew = x.shape
        x = self.bnrelu(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x).squeeze(1))
        x = self.conv4(x)
        x = F.interpolate(x, [int(featureh * 4), int(featurew * 4)], mode='bilinear', align_corners=False)
        pred = self.softmax(x)
        return pred

class EppFlowNet_Decoder(nn.Module):
    def __init__(self):
        super(EppFlowNet_Decoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=64, out_channels=48, kernel_size=3, padding=1)
        self.conv2 = Conv3d(dimin=48, dimout=48)
        self.conv3 = Conv3d(dimin=48, dimout=48)

        self.hug3d1 = Hourglass3D(n=4, dimin=48)
        self.hug3d2 = Hourglass3D(n=4, dimin=48)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv3(self.conv2(x))

        d_feature1 = self.hug3d1(x)
        d_feature2 = self.hug3d2(d_feature1)

        return d_feature1, d_feature2

class EppFlowNet(nn.Module):
    def __init__(self, args):
        super(EppFlowNet, self).__init__()
        self.args = args
        self.nedges = self.args.num_scales * self.args.num_angs

        self.encorder = EppFlowNet_Encoder()
        self.decoder = EppFlowNet_Decoder()
        self.scalehead = Scalehead(self.args)

        self.pts2ddict = dict()

        self.eppinflate = eppcore_inflation.apply
        self.eppcompress = eppcore_compression.apply

        self.init_ang2RM()

    def init_ang2RM(self):
        ang2RMx = torch.zeros([3, 3, 6])
        ang2RMy = torch.zeros([3, 3, 6])
        ang2RMz = torch.zeros([3, 3, 6])

        ang2RMx[1, 1, 0] = 1
        ang2RMx[1, 2, 0 + 3] = -1
        ang2RMx[2, 1, 0 + 3] = 1
        ang2RMx[2, 2, 0] = 1

        ang2RMy[0, 0, 1] = 1
        ang2RMy[0, 2, 1 + 3] = 1
        ang2RMy[2, 0, 1 + 3] = -1
        ang2RMy[2, 2, 1] = 1

        ang2RMz[0, 0, 2] = 1
        ang2RMz[0, 1, 2 + 3] = -1
        ang2RMz[1, 0, 2 + 3] = 1
        ang2RMz[1, 1, 2] = 1

        expnum = int(self.args.num_angs)
        ang2RMx = ang2RMx.view([1, 1, 1, 3, 3, 1, 6]).expand([-1, expnum, self.args.maxinsnum, -1, -1, -1, -1])
        self.ang2RMx = torch.nn.Parameter(ang2RMx, requires_grad=False)
        ang2RMy = ang2RMy.view([1, 1, 1, 3, 3, 1, 6]).expand([1, expnum, self.args.maxinsnum, -1, -1, -1, -1])
        self.ang2RMy = torch.nn.Parameter(ang2RMy, requires_grad=False)
        ang2RMz = ang2RMz.view([1, 1, 1, 3, 3, 1, 6]).expand([1, expnum, self.args.maxinsnum, -1, -1, -1, -1])
        self.ang2RMz = torch.nn.Parameter(ang2RMz, requires_grad=False)

        rotxcomp = torch.zeros([3, 3])
        rotxcomp[0, 0] = 1
        rotxcomp = rotxcomp.view([1, 1, 1, 3, 3]).expand([1, expnum, self.args.maxinsnum, -1, -1])
        self.rotxcomp = torch.nn.Parameter(rotxcomp, requires_grad=False)

        rotycomp = torch.zeros([3, 3])
        rotycomp[1, 1] = 1
        rotycomp = rotycomp.view([1, 1, 1, 3, 3]).expand([1, expnum, self.args.maxinsnum, -1, -1])
        self.rotycomp = torch.nn.Parameter(rotycomp, requires_grad=False)

        rotzcomp = torch.zeros([3, 3])
        rotzcomp[2, 2] = 1
        rotzcomp = rotzcomp.view([1, 1, 1, 3, 3]).expand([1, expnum, self.args.maxinsnum, -1, -1])
        self.rotzcomp = torch.nn.Parameter(rotzcomp, requires_grad=False)

        lastrowpad = torch.zeros([1, 4])
        lastrowpad[0, 3] = 1
        lastrowpad = lastrowpad.view([1, 1, 1, 1, 4]).expand([1, expnum, self.args.maxinsnum, -1, -1])
        self.lastrowpad = torch.nn.Parameter(lastrowpad, requires_grad=False)

    def ang2R(self, angs):
        """Convert an axisangle rotation into a 4x4 transformation matrix
        (adapted from https://github.com/Wallacoloo/printipi)
        Input 'vec' has to be Bx1x3
        """
        expnum = int(self.args.num_angs)
        bz = angs.shape[0]
        cos_sin = torch.cat([torch.cos(angs), torch.sin(angs)], dim=-1).view([bz, expnum, self.args.maxinsnum, 1, 1, 6, 1]).expand([-1, -1, -1, 3, 3, -1, -1])
        rotx = (self.ang2RMx @ cos_sin).squeeze(-1).squeeze(-1) + self.rotxcomp.expand([bz, -1, -1, -1, -1])
        roty = (self.ang2RMy @ cos_sin).squeeze(-1).squeeze(-1) + self.rotycomp.expand([bz, -1, -1, -1, -1])
        rotz = (self.ang2RMz @ cos_sin).squeeze(-1).squeeze(-1) + self.rotzcomp.expand([bz, -1, -1, -1, -1])
        rot = rotz @ roty @ rotx
        return rot

    def resample_feature(self, feature1, feature2, depthpred, intrinsic, posepred, insmap, orgh, orgw):
        bz, featurc, featureh, featurew = feature1.shape
        dsratio = int(orgh / featureh)
        sample_pts, inboundmask = self.get_samplecoords(depthpred, intrinsic, posepred, insmap, dsratio, orgh, orgw)

        feature2_ex = feature2.unsqueeze(1).expand([-1, self.nedges, -1, -1, -1]).contiguous().view([bz * self.nedges, featurc, featureh, featurew])
        sampled_feature2 = F.grid_sample(feature2_ex, sample_pts, mode='bilinear', align_corners=False).view([bz, self.nedges, featurc, featureh, featurew]).permute([0, 2, 1, 3, 4])
        feature_volume = torch.cat([feature1.unsqueeze(2).expand([-1, -1, self.nedges, -1, -1]), sampled_feature2], dim=1)
        sample_pts = sample_pts.view(bz, self.nedges, featureh, featurew, 2)
        return feature_volume, sample_pts, inboundmask

    def get_samplecoords(self, depthpred, intrinsic, posepred, insmap, dsratio, orgh, orgw):
        featureh = int(orgh / dsratio)
        featurew = int(orgw / dsratio)
        bz = intrinsic.shape[0]

        intrinsicex = intrinsic.unsqueeze(1).unsqueeze(1).expand([-1, self.nedges, self.args.maxinsnum, -1, -1])
        projM = intrinsicex @ posepred @ torch.inverse(intrinsicex)
        insmap_ex = insmap.expand([-1, self.nedges, -1, -1])
        projMimg = self.eppinflate(insmap_ex.contiguous().view([-1, 1, orgh, orgw]).contiguous(), projM.view([-1, self.args.maxinsnum,  4, 4]).contiguous())
        projMimg = projMimg.view([bz, self.nedges, orgh, orgw, 4, 4])

        sample_depthmap = depthpred.expand([-1, self.nedges, -1, -1])

        infkey = "{}_{}_{}".format(bz, orgh, orgw)
        if infkey not in self.pts2ddict.keys():
            xx, yy = np.meshgrid(range(orgw), range(orgh), indexing='xy')
            xx = torch.from_numpy(xx).float().unsqueeze(0).unsqueeze(0).expand([bz, -1, -1, -1]).cuda(depthpred.device)
            yy = torch.from_numpy(yy).float().unsqueeze(0).unsqueeze(0).expand([bz, -1, -1, -1]).cuda(depthpred.device)
            ones = torch.ones_like(xx)
            self.pts2ddict[infkey] = (xx, yy, ones)
        xx, yy, ones = self.pts2ddict[infkey]

        xx = xx.expand([-1, self.nedges, -1, -1])
        yy = yy.expand([-1, self.nedges, -1, -1])
        ones = ones.expand([-1, self.nedges, -1, -1])

        pts3d = torch.stack([xx * sample_depthmap, yy * sample_depthmap, sample_depthmap, ones], dim=-1).unsqueeze(-1)
        pts2dp = projMimg @ pts3d

        pxx, pyy, pzz, _ = torch.split(pts2dp, 1, dim=4)

        sign = pzz.sign()
        sign[sign == 0] = 1
        pzz = torch.clamp(torch.abs(pzz), min=1e-20) * sign

        pxx = (pxx / pzz).squeeze(-1).squeeze(-1)
        pyy = (pyy / pzz).squeeze(-1).squeeze(-1)

        supressval = torch.ones_like(pxx) * (-100)

        inboundmask = ((pzz > 1e-20).squeeze(-1).squeeze(-1) * (pxx >= 0) * (pyy >= 0) * (pxx < orgw) * (pyy < orgh)).float()

        pxx = inboundmask * pxx + supressval * (1 - inboundmask)
        pyy = inboundmask * pyy + supressval * (1 - inboundmask)

        sample_px = pxx / float(dsratio)
        sample_px = F.interpolate(sample_px, [featureh, featurew], mode='nearest')
        sample_px = (sample_px / featurew - 0.5) * 2

        sample_py = pyy / float(dsratio)
        sample_py = F.interpolate(sample_py, [featureh, featurew], mode='nearest')
        sample_py = (sample_py / featureh - 0.5) * 2

        sample_pts = torch.stack([sample_px, sample_py], dim=-1).view(bz * self.nedges, featureh, featurew, 2)
        return sample_pts, inboundmask

    def ang_t_to_aff(self, ang, t):
        rot = self.ang2R(ang)
        afft = torch.cat([rot, t.unsqueeze(-1)], dim=-1)
        afft = torch.cat([afft, self.lastrowpad.expand([ang.shape[0], -1, -1, -1, -1])], dim=3)
        return afft

    def adjustpose(self, d_feature, angin, scalein, mvdin, insmap, inboundmask, orgh, orgw, k, posepred=None):
        bz = insmap.shape[0]

        insmap_cp = torch.clone(insmap)
        insmap_manip_selector = torch.ones_like(insmap)
        insmap_manip_selector[:, :, int(0.25810811 * orgh): int(0.99189189 * orgh)] = 0
        insmap_manip_selector[(insmap_cp == 0) * (inboundmask[:, 5].unsqueeze(1) == 0)] = 1
        insmap_manip_selector = insmap_manip_selector == 1
        # from core.utils.utils import tensor2disp
        # tensor2disp(insmap_manip_selector, viewind=0, vmax=1).show()
        insmap_cp[insmap_manip_selector] = -1
        insmap_num = self.eppcompress(insmap_cp, torch.ones([bz, orgh, orgw, 1, 1], dtype=torch.float, device=insmap.device), self.args.maxinsnum) + 1e-6


        rss = self.scalehead(d_feature)
        rss = self.eppcompress(insmap_cp, rss.permute([0, 2, 3, 1]).unsqueeze(-1).contiguous(), self.args.maxinsnum) / insmap_num
        rss = rss.permute([0, 2, 1, 3])
        scale_adj = scalein * torch.exp(rss)

        mvdout = scale_adj.expand([-1, -1, -1, 3]) * mvdin
        angout = angin

        afft_all = self.ang_t_to_aff(ang=angout, t=mvdout)

        # Validation
        # if posepred is not None:
        #     rndbz = np.random.randint(0, bz)
        #     rndang = np.random.randint(0, self.args.num_angs)
        #     refpose = posepred[rndbz, rndang * self.args.num_scales + 5, 0]
        #     valpose = afft_all[rndbz, rndang, 0]
        #     print(torch.sum(torch.abs(refpose - valpose)))
        outputs = dict()
        outputs[('afft_all', k)] = afft_all
        outputs[('scale_adj', k)] = scale_adj
        return outputs

    def forward(self, image1, image2, depthpred, intrinsic, posepred, angin, scalein, mvdin, insmap):
        """ Estimate optical flow between pair of frames """
        _, _, orgh, orgw = image1.shape
        image1_normed = 2 * image1 - 1.0
        image2_normed = 2 * image2 - 1.0

        feature1 = self.encorder(image1_normed)
        feature2 = self.encorder(image2_normed)

        feature_volume, sample_pts, inboundmask = self.resample_feature(feature1, feature2, depthpred, intrinsic, posepred, insmap, orgh, orgw)

        d_feature1, d_feature2 = self.decoder(feature_volume)

        outputs = dict()
        outputs.update(self.adjustpose(d_feature1, angin, scalein, mvdin, insmap, inboundmask, orgh, orgw, k=1, posepred=posepred))
        outputs.update(self.depth2rgb(depthpred, image2, outputs, insmap, intrinsic, k=1))
        outputs.update(self.adjustpose(d_feature2, angin, scalein, mvdin, insmap, inboundmask, orgh, orgw, k=2, posepred=posepred))
        outputs.update(self.depth2rgb(depthpred, image2, outputs, insmap, intrinsic, k=2))
        outputs['sample_pts'] = sample_pts
        return outputs

    def depth2rldepth(self, depthpred, objscale, insmap, outputs):
        objscale_inf = self.eppinflate(insmap, objscale).squeeze(-1).squeeze(-1).unsqueeze(1)
        for k in range(1, 3, 1):
            outputs[('org_relativedepth', k)] = torch.log(depthpred + 1e-10) - objscale_inf
            outputs[('relativedepth', k)] = outputs[('org_relativedepth', k)] + outputs[('residualdepth', k)]
        return outputs

    def depth2flow(self, depth, projMimg):
        bz, _, orgh, orgw = depth.shape
        infkey = "{}_{}_{}".format(bz, orgh, orgw)
        xx, yy, ones = self.pts2ddict[infkey]

        pts3d = torch.stack([xx * depth, yy * depth, depth, ones], dim=-1).squeeze(1).unsqueeze(-1)
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

    def depth2rgb(self, depth, img2, outputs, insmap, intrinsic, k):
        bz, _, orgh, orgw = depth.shape
        infkey = "{}_{}_{}".format(bz, orgh, orgw)
        xx, yy, ones = self.pts2ddict[infkey]

        expnum = int(self.args.num_angs)

        pts3d = torch.stack([xx * depth, yy * depth, depth, ones], dim=-1).unsqueeze(-1).expand([-1, expnum, -1, -1, -1, -1])
        poses = outputs[('afft_all', k)]

        intrinsicex = intrinsic.unsqueeze(1).unsqueeze(1).expand([-1, expnum, self.args.maxinsnum, -1, -1])
        projM = intrinsicex @ poses @ torch.inverse(intrinsicex)
        insmap_ex = insmap.expand([-1, expnum, -1, -1])
        projMimg = self.eppinflate(insmap_ex.contiguous().view([-1, 1, orgh, orgw]).contiguous(), projM.view([-1, self.args.maxinsnum,  4, 4]).contiguous())
        projMimg = projMimg.view([bz, expnum, orgh, orgw, 4, 4])

        pts2dp = projMimg @ pts3d

        pxx, pyy, pzz, _ = torch.split(pts2dp, 1, dim=4)

        sign = pzz.sign()
        sign[sign == 0] = 1
        pzz = torch.clamp(torch.abs(pzz), min=1e-20) * sign

        pxx = (pxx / pzz).squeeze(-1).squeeze(-1)
        pyy = (pyy / pzz).squeeze(-1).squeeze(-1)

        supressval = torch.ones_like(pxx) * (-100)
        inboundmask = ((pzz > 1e-20).squeeze(-1).squeeze(-1)).float()

        pxx = inboundmask * pxx + supressval * (1 - inboundmask)
        pyy = inboundmask * pyy + supressval * (1 - inboundmask)

        pxx = (pxx / orgw - 0.5) * 2
        pyy = (pyy / orgh - 0.5) * 2

        samplepts = torch.stack([pxx, pyy], dim=-1)
        img1_recon = F.grid_sample(img2.unsqueeze(1).expand([-1, expnum, -1, -1, -1]).contiguous().view(-1, 3, orgh, orgw), samplepts.view(-1, orgh, orgw, 2), mode='bilinear', align_corners=False)
        img1_recon = img1_recon.view(bz, expnum, 3, orgh, orgw)

        # from core.utils.utils import tensor2disp, tensor2rgb
        # tensor2rgb(img1_recon[:, 0], viewind=0).show()
        # tensor2rgb(img1_recon[:, -1], viewind=0).show()
        # tensor2rgb(img1_recon[:, 0], viewind=1).show()
        outputs = dict()
        outputs[('img1_recon', k)] = img1_recon
        return outputs