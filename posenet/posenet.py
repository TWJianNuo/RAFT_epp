# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch.nn.functional as F
from eppcore import eppcore_inflation, eppcore_compression

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, args, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.args = args

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 7 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

        angrange = np.expand_dims(np.stack([args.angx_range, args.angy_range, args.angz_range], axis=0), axis=0)
        self.angrange = nn.Parameter(torch.from_numpy(angrange).float(), requires_grad=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        ang, tdir, tscale = torch.split(out, [3, 3, 1], dim=1)
        tdir = tdir / torch.sqrt(torch.sum(tdir ** 2, dim=1, keepdim=True))

        tscale = self.sigmoid(tscale) * self.args.tscale_range
        ang = (self.sigmoid(ang) - 0.5) * 2 * self.angrange.expand([ang.shape[0], -1])
        return ang, tdir, tscale

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

class ObjPoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, args, scales=range(4), num_output_channels=1, use_skips=True):
        super(ObjPoseDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.args = args

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("obj_scale", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            self.convs[("obj_angle", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("obj_scale", i)] = self.sigmoid(self.convs[("obj_scale", i)](x)) * self.args.objtscale_range
                self.outputs[("obj_angle", i)] = (self.sigmoid(self.convs[("obj_angle", i)](x)) - 0.5) * 2 * np.pi * 2

        return self.outputs

class Posenet(nn.Module):
    def __init__(self, num_layers, args):
        super(Posenet, self).__init__()
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=True, num_input_images=2)
        self.self_pose_decoder = PoseDecoder(num_ch_enc=self.encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, args=args)
        self.obj_pose_decoder = ObjPoseDecoder(num_ch_enc=self.encoder.num_ch_enc, num_output_channels=1, args=args)
        self.args = args

        self.eppinflate = eppcore_inflation.apply
        self.eppcompress = eppcore_compression.apply

        self.init_ang2RM_obj()
        self.init_ang2RM_self()
        self.init_pts2d()

    def init_pts2d(self):
        xx, yy = np.meshgrid(range(self.args.inwidth), range(self.args.inheight), indexing='xy')
        pts2d = np.stack([xx, yy, np.ones_like(xx)], axis=2)
        self.pts2d = nn.Parameter(torch.from_numpy(pts2d).float().view([1, self.args.inheight, self.args.inwidth, 3, 1]), requires_grad=False)

    def init_ang2RM_obj(self):
        ang2RM_obj = torch.zeros([4, 4, 1, 2])

        ang2RM_obj[0, 3, 0, 0] = 1
        ang2RM_obj[2, 3, 0, 1] = -1

        ang2RM_obj = ang2RM_obj.view([1, 1, 4, 4, 1, 2])
        self.ang2RM_obj = torch.nn.Parameter(ang2RM_obj, requires_grad=False)

        rotcomp_obj = torch.zeros([4, 4])
        rotcomp_obj[0, 0] = 1
        rotcomp_obj[1, 1] = 1
        rotcomp_obj[2, 2] = 1
        rotcomp_obj[3, 3] = 1
        rotcomp_obj = rotcomp_obj.view([1, 1, 4, 4])
        self.rotcomp_obj = torch.nn.Parameter(rotcomp_obj, requires_grad=False)

    def mvinfo2objpose(self, objang, objscale, selfpose):
        bz, maxins, _, _ = objang.shape
        cos_sin = torch.cat([torch.cos(objang), torch.sin(objang)], dim=2).view([bz, maxins, 1, 1, 2, 1]).expand([-1, -1, 4, 4, -1, -1])

        ang2RM_obj = self.ang2RM_obj.expand([bz, maxins, -1, -1, -1, -1])
        rotcomp_obj = self.rotcomp_obj.expand([bz, maxins, -1, -1])

        M = (ang2RM_obj @ cos_sin).squeeze(-1).squeeze(-1) * objscale + rotcomp_obj
        objpose = M @ selfpose.unsqueeze(1).expand([-1, maxins, -1, -1])

        if maxins > 1:
            _, robjposes = torch.split(objpose, [1, maxins-1], dim=1)
            poses = torch.cat([selfpose.unsqueeze(1), robjposes], dim=1)
        else:
            poses = objpose * 0 + selfpose.unsqueeze(1)
        return poses

    def init_ang2RM_self(self):
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

        ang2RMx = ang2RMx.view([1, 3, 3, 1, 6])
        self.ang2RMx = torch.nn.Parameter(ang2RMx, requires_grad=False)
        ang2RMy = ang2RMy.view([1, 3, 3, 1, 6])
        self.ang2RMy = torch.nn.Parameter(ang2RMy, requires_grad=False)
        ang2RMz = ang2RMz.view([1, 3, 3, 1, 6])
        self.ang2RMz = torch.nn.Parameter(ang2RMz, requires_grad=False)

        rotxcomp = torch.zeros([3, 3])
        rotxcomp[0, 0] = 1
        rotxcomp = rotxcomp.view([1, 3, 3])
        self.rotxcomp = torch.nn.Parameter(rotxcomp, requires_grad=False)

        rotycomp = torch.zeros([3, 3])
        rotycomp[1, 1] = 1
        rotycomp = rotycomp.view([1, 3, 3])
        self.rotycomp = torch.nn.Parameter(rotycomp, requires_grad=False)

        rotzcomp = torch.zeros([3, 3])
        rotzcomp[2, 2] = 1
        rotzcomp = rotzcomp.view([1, 3, 3])
        self.rotzcomp = torch.nn.Parameter(rotzcomp, requires_grad=False)

    def get_selfpose(self, selfang, selftdir, selfscale):
        bz, _ = selfang.shape
        cos_sin = torch.cat([torch.cos(selfang), torch.sin(selfang)], dim=1).view([bz, 1, 1, 6, 1]).expand([bz, 3, 3, -1, -1])
        rotx = (self.ang2RMx @ cos_sin).squeeze(-1).squeeze(-1) + self.rotxcomp
        roty = (self.ang2RMy @ cos_sin).squeeze(-1).squeeze(-1) + self.rotycomp
        rotz = (self.ang2RMz @ cos_sin).squeeze(-1).squeeze(-1) + self.rotzcomp
        selfR = rotz @ roty @ rotx
        selfT = (selftdir * selfscale).unsqueeze(-1)

        selfRT = torch.eye(4, device=selfang.device)
        selfRT = selfRT.view([1, 4, 4]).repeat([bz, 1, 1])
        selfRT[:, 0:3, :] = torch.cat([selfR, selfT], dim=2)
        return selfRT

    def depth2flow(self, depthmap, instance, intrinsic, t, R, pts2d=None):
        bz, _, h, w = depthmap.shape

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

        numx = (m1 @ pts2d).view(bz, 1, h, w) * depthmap
        numy = (m2 @ pts2d).view(bz, 1, h, w) * depthmap
        denom = (m3 @ pts2d).view(bz, 1, h, w) * depthmap

        t1 = t1.view(bz, 1, h, w)
        t2 = t2.view(bz, 1, h, w)
        t3 = t3.view(bz, 1, h, w)

        denomf = (denom + t3)
        sign = denomf.sign()
        sign[sign == 0] = 1
        denomf = torch.clamp(torch.abs(denomf), min=1e-20) * sign

        px = (numx + t1) / denomf
        py = (numy + t2) / denomf

        flowx = px.squeeze(1) - self.pts2d[:, :, :, 0, 0]
        flowy = py.squeeze(1) - self.pts2d[:, :, :, 1, 0]

        flowpred = torch.stack([flowx, flowy], dim=1)
        return flowpred

    def forward(self, img1, img2):
        img = torch.cat([img1, img2], dim=1)

        features = self.encoder(img)

        self_ang, self_tdir, self_tscale = self.self_pose_decoder([features])
        obj_pose = self.obj_pose_decoder(features)

        return self_ang, self_tdir, self_tscale, obj_pose