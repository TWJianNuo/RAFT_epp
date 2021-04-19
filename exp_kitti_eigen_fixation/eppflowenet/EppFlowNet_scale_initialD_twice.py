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

class StereoHead(nn.Module):
    def __init__(self, args):
        super(StereoHead, self).__init__()
        self.bnrelu = BnRelu3d(48)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(in_channels=48, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.args = args

    def forward(self, x):
        _, _, _, featureh, featurew = x.shape
        x = self.bnrelu(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x).squeeze(1))
        x = self.conv4(x)
        x = F.interpolate(x, [int(featureh * 4), int(featurew * 4)], mode='bilinear', align_corners=False)
        pred = (self.sigmoid(x) - 0.5) * 2 * self.args.maxlogscale
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
        self.nedges = self.args.num_deges
        self.encorder = EppFlowNet_Encoder()
        self.decoder = EppFlowNet_Decoder()
        self.stereohead = StereoHead(self.args)

        # Sample absolute depth
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'depth_bin.pickle'), 'rb') as f:
            linlogdedge = pickle.load(f)

        assert self.args.num_deges == linlogdedge.shape[0]
        self.reld_edges = nn.Parameter(torch.from_numpy(linlogdedge).float().view([1, self.args.num_deges, 1, 1]), requires_grad=False)
        self.pts2ddict = dict()

        self.eppinflate = eppcore_inflation.apply
        self.eppcompress = eppcore_compression.apply

    def resample_feature(self, feature1, feature2, depthpred, intrinsic, posepred, insmap, orgh, orgw):
        bz, featurc, featureh, featurew = feature1.shape
        dsratio = int(orgh / featureh)
        sample_pts, projMimg = self.get_samplecoords(depthpred, intrinsic, posepred, insmap, dsratio, orgh, orgw)

        feature2_ex = feature2.unsqueeze(1).expand([-1, self.nedges, -1, -1, -1]).contiguous().view([bz * self.nedges, featurc, featureh, featurew])
        sampled_feature2 = F.grid_sample(feature2_ex, sample_pts, mode='bilinear', align_corners=False).view([bz, self.nedges, featurc, featureh, featurew]).permute([0, 2, 1, 3, 4])
        feature_volume = torch.cat([feature1.unsqueeze(2).expand([-1, -1, self.nedges, -1, -1]), sampled_feature2], dim=1)
        return feature_volume, sample_pts, projMimg

    def get_samplecoords(self, depthpred, intrinsic, posepred, insmap, dsratio, orgh, orgw):
        featureh = int(orgh / dsratio)
        featurew = int(orgw / dsratio)
        bz = intrinsic.shape[0]

        intrinsicex = intrinsic.unsqueeze(1).expand([-1, self.args.maxinsnum, -1, -1])
        projM = intrinsicex @ posepred @ torch.inverse(intrinsicex)
        projMimg = self.eppinflate(insmap, projM)

        reld_edges = self.reld_edges.expand([bz, -1, orgh, orgw])
        sample_depthmap = torch.exp(torch.log(depthpred.expand([-1, self.nedges, -1, -1])) + reld_edges)

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
        pts2dp = projMimg.unsqueeze(1).expand([-1, self.nedges, -1, -1, -1, -1]) @ pts3d

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
        return sample_pts, projMimg

    def forward(self, image1, image2, depthpred, intrinsic, posepred, insmap):
        """ Estimate optical flow between pair of frames """
        _, _, orgh, orgw = image1.shape
        image1_normed = 2 * image1 - 1.0
        image2_normed = 2 * image2 - 1.0

        feature1 = self.encorder(image1_normed)
        feature2 = self.encorder(image2_normed)

        outputs = dict()

        feature_volume, sample_pts, projMimg = self.resample_feature(feature1, feature2, depthpred, intrinsic, posepred, insmap, orgh, orgw)

        d_feature1, d_feature2 = self.decoder(feature_volume)
        residual_depth1 = self.stereohead(d_feature1)
        residual_depth2 = self.stereohead(d_feature2)

        depth1 = torch.exp(torch.log(depthpred) + residual_depth1)
        depth2 = torch.exp(torch.log(depthpred) + residual_depth2)

        depthpred = depth2
        feature_volume, sample_pts, projMimg = self.resample_feature(feature1, feature2, depthpred, intrinsic, posepred, insmap, orgh, orgw)

        d_feature1, d_feature2 = self.decoder(feature_volume)
        residual_depth1 = self.stereohead(d_feature1)
        residual_depth2 = self.stereohead(d_feature2)

        depth3 = torch.exp(torch.log(depthpred) + residual_depth1)
        depth4 = torch.exp(torch.log(depthpred) + residual_depth2)

        outputs[('depth', 1)] = depth1
        outputs[('depth', 2)] = depth2
        outputs[('depth', 3)] = depth3
        outputs[('depth', 4)] = depth4
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

    def depth2rgb(self, depth, projMimg, img2):
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
        return flowpred, img1_recon