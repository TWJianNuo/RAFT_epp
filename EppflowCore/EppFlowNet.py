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
        self.softmax = nn.Softmax(dim=1)
        self.args = args

    def soft_argmax(self, x, reld_edges):
        """ Convert probability volume into point estimate of depth"""
        x = self.softmax(x)
        bz, _, h, w = x.shape
        pred = torch.sum(x * reld_edges.expand([bz, -1, h, w]), dim=1, keepdim=True)
        return pred

    def forward(self, x, reld_edges):
        x = self.bnrelu(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x).squeeze(1)
        x = F.interpolate(x, [self.args.inheight, self.args.inwidth], mode='bilinear', align_corners=False)
        pred = self.soft_argmax(x, reld_edges)
        return pred

class ScaleHead(nn.Module):
    def __init__(self, args):
        super(ScaleHead, self).__init__()
        self.bnrelu = BnRelu(48)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.args = args

    def forward(self, x, instance, insnum, eppinflate, eppcompress):
        x = self.bnrelu(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        pred = self.sigmoid(x) * self.args.maxscale + 1e-2
        pred = F.interpolate(pred, [self.args.inheight, self.args.inwidth], mode='bilinear', align_corners=False)

        pred_compressed = eppcompress(instance, pred.squeeze(1).unsqueeze(-1).unsqueeze(-1), self.args.maxinsnum)
        pred_compressed = pred_compressed / (insnum + 1e-3)
        pred_inf = eppinflate(instance, pred_compressed).squeeze(-1).squeeze(-1).unsqueeze(1)
        pred_inf = torch.clamp(pred_inf, 1e-2)
        return pred, pred_inf

class EppFlowNet_Decoder(nn.Module):
    def __init__(self):
        super(EppFlowNet_Decoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=64, out_channels=48, kernel_size=3, padding=1)
        self.conv2 = Conv3d(dimin=48, dimout=48)
        self.conv3 = Conv3d(dimin=48, dimout=48)

        self.hug3d1 = Hourglass3D(n=4, dimin=48)
        self.hug3d2 = Hourglass3D(n=4, dimin=48)

        self.conv4 = Conv2d(dimin=32, dimout=1)
        self.hug2d1 = Hourglass2D(n=4, dimin=48)
        self.hug2d2 = Hourglass2D(n=4, dimin=48)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv3(self.conv2(x))

        d_feature1 = self.hug3d1(x)
        d_feature2 = self.hug3d2(d_feature1)

        bz, nfeature, ndepth, h, w = x.shape
        s = self.conv4(x.view(bz * nfeature, ndepth, h, w)).view(bz, nfeature, h, w)
        s_feature1 = self.hug2d1(s)
        s_feature2 = self.hug2d2(s)

        return d_feature1, d_feature2, s_feature1, s_feature2

class EppFlowNet(nn.Module):
    def __init__(self, args):
        super(EppFlowNet, self).__init__()
        self.args = args
        self.encorder = EppFlowNet_Encoder()
        self.decoder = EppFlowNet_Decoder()
        self.stereohead = StereoHead(self.args)
        self.scalehead = ScaleHead(self.args)

        reld_edges_pkl = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'depth_bin.pickle')
        reld_edges = pickle.load(open(reld_edges_pkl, "rb"))
        self.reld_edges = nn.Parameter(torch.from_numpy(reld_edges).float().view([1, reld_edges.shape[0], 1, 1]), requires_grad=False)
        self.nedges = reld_edges.shape[0]

        featureh = int(self.args.inheight / 4)
        featurew = int(self.args.inwidth / 4)
        xx, yy = np.meshgrid(range(featurew), range(featureh), indexing='xy')
        pts2d = np.stack([xx, yy, np.ones_like(xx)], axis=2)
        self.pts2d = nn.Parameter(torch.from_numpy(pts2d).float().view([1, featureh, featurew, 3, 1]), requires_grad=False)

        self.eppinflate = eppcore_inflation.apply
        self.eppcompress = eppcore_compression.apply

    def resample_feature(self, feature1, feature2, instance, intrinsic, t, R, pts2d):
        bz, featurc, featureh, featurew = feature1.shape

        reld_edges = self.reld_edges.expand([bz, -1, featureh, featurew])
        if pts2d is None:
            pts2d = self.pts2d
        pts2d = pts2d.expand([bz, -1, -1, -1, -1])

        intrinsic_ex = intrinsic.view(bz, 1, 3, 3).expand([-1, self.args.maxinsnum, -1, -1])
        M = intrinsic_ex @ R @ torch.inverse(intrinsic_ex)
        T = intrinsic_ex @ t

        M_inf = self.eppinflate(instance, M)
        T_inf = self.eppinflate(instance, T)

        m1, m2, m3 = torch.split(M_inf, 1, dim=3)
        t1, t2, t3 = torch.split(T_inf, 1, dim=3)

        numx = (m1 @ pts2d).view(bz, 1, featureh, featurew).expand([-1, self.nedges, -1, -1])
        numy = (m2 @ pts2d).view(bz, 1, featureh, featurew).expand([-1, self.nedges, -1, -1])
        denom = (m3 @ pts2d).view(bz, 1, featureh, featurew).expand([-1, self.nedges, -1, -1])

        t1 = t1.view(bz, 1, featureh, featurew).expand([-1, self.nedges, -1, -1])
        t2 = t2.view(bz, 1, featureh, featurew).expand([-1, self.nedges, -1, -1])
        t3 = t3.view(bz, 1, featureh, featurew).expand([-1, self.nedges, -1, -1])

        denomf = (denom + reld_edges * t3)
        sign = denomf.sign()
        sign[sign == 0] = 1
        denomf = torch.clamp(torch.abs(denomf), min=1e-20) * sign

        sample_px = ((numx + reld_edges * t1) / denomf / featurew - 0.5) * 2
        sample_py = ((numy + reld_edges * t2) / denomf / featureh - 0.5) * 2
        sample_pts = torch.stack([sample_px, sample_py], dim=-1).view(bz * self.nedges, featureh, featurew, 2)
        feature2_ex = feature2.unsqueeze(1).expand([-1, self.nedges, -1, -1, -1]).view([bz * self.nedges, featurc, featureh, featurew])
        sampled_feature2 = F.grid_sample(feature2_ex, sample_pts, mode='bilinear', align_corners=False).view([bz, self.nedges, featurc, featureh, featurew]).permute([0, 2, 1, 3, 4])
        feature_volume = torch.cat([feature1.unsqueeze(2).expand([-1, -1, self.nedges, -1, -1]), sampled_feature2], dim=1)
        return feature_volume

    def forward(self, image1, image2, instance, instance_featuresize, intrinsic, t, R, pts2d=None):
        """ Estimate optical flow between pair of frames """
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        feature1 = self.encorder(image1)
        feature2 = self.encorder(image2)

        feature_volume = self.resample_feature(feature1, feature2, instance_featuresize, intrinsic, t, R, pts2d)

        d_feature1, d_feature2, s_feature1, s_feature2 = self.decoder(feature_volume)
        eppflow1 = self.stereohead(d_feature1, self.reld_edges)
        eppflow2 = self.stereohead(d_feature2, self.reld_edges)

        insnum = self.eppcompress(instance, (instance > -1).float().squeeze(1).unsqueeze(-1).unsqueeze(-1), self.args.maxinsnum)
        scale1, scale1_inf = self.scalehead(s_feature1, instance, insnum, self.eppinflate, self.eppcompress)
        scale2, scale2_inf = self.scalehead(s_feature2, instance, insnum, self.eppinflate, self.eppcompress)

        return eppflow1, eppflow2, scale1, scale2

    def validate_sample(self, image1, image2, depthmap, reldepthmap, flowmap, instance, intrinsic, t, R, pts2d):
        bz, _, h, w = image1.shape
        pts2d = pts2d.expand([bz, -1, -1, -1, -1])

        intrinsic_ex = intrinsic.view(bz, 1, 3, 3).expand([-1, self.args.maxinsnum, -1, -1])
        M = intrinsic_ex @ R @ torch.inverse(intrinsic_ex)
        T = intrinsic_ex @ t

        M_inf = self.eppinflate(instance, M)
        T_inf = self.eppinflate(instance, T)

        m1, m2, m3 = torch.split(M_inf, 1, dim=3)
        t1, t2, t3 = torch.split(T_inf, 1, dim=3)

        numx = (m1 @ pts2d).view(bz, 1, h, w)
        numy = (m2 @ pts2d).view(bz, 1, h, w)
        denom = (m3 @ pts2d).view(bz, 1, h, w)

        t1 = t1.view(bz, 1, h, w)
        t2 = t2.view(bz, 1, h, w)
        t3 = t3.view(bz, 1, h, w)

        denomf = (denom + reldepthmap * t3)
        sign = denomf.sign()
        sign[sign == 0] = 1
        denomf = torch.clamp(torch.abs(denomf), min=1e-20) * sign

        px = (numx + reldepthmap * t1) / denomf
        py = (numy + reldepthmap * t2) / denomf

        import matplotlib.pyplot as plt
        cm = plt.get_cmap('magma')
        vmax = 0.15

        pxnp = px.squeeze().cpu().numpy()
        pynp = py.squeeze().cpu().numpy()
        depthmapnp = depthmap.squeeze().cpu().numpy()

        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        objsample = 1000
        orgpts = list()
        flowpts = list()
        colors = list()
        for k in torch.unique(instance).cpu().numpy():
            obj_sel = (instance == k).squeeze().cpu().numpy()

            rndidx = np.random.randint(0, np.sum(obj_sel), objsample)

            objxxf = xx[obj_sel][rndidx]
            objyyf = yy[obj_sel][rndidx]

            objdf = depthmapnp[objyyf, objxxf]
            objcolor = 1 / objdf / vmax
            objcolor = cm(objcolor)

            objxxf_o = pxnp[objyyf, objxxf]
            objyyf_o = pynp[objyyf, objxxf]

            orgpts.append(np.stack([objxxf, objyyf], axis=0))
            flowpts.append(np.stack([objxxf_o, objyyf_o], axis=0))
            colors.append(objcolor)

        image1np = image1[0].cpu().permute([1, 2, 0]).numpy().astype(np.uint8)
        image2np = image2[0].cpu().permute([1, 2, 0]).numpy().astype(np.uint8)

        fig = plt.figure(figsize=(16, 9))
        fig.add_subplot(2, 1, 1)
        for k in range(torch.unique(instance).cpu().numpy().shape[0]):
            plt.scatter(orgpts[k][0], orgpts[k][1], 1, colors[k])
        plt.imshow(image1np)

        fig.add_subplot(2, 1, 2)
        for k in range(torch.unique(instance).cpu().numpy().shape[0]):
            plt.scatter(flowpts[k][0], flowpts[k][1], 1, colors[k])
        plt.imshow(image2np)

        plt.show()



