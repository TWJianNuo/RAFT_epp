from EppflowLEAStereo.models.build_model_2d import Disp
from EppflowLEAStereo.models.decoding_formulas import network_layer_to_space
from EppflowLEAStereo.retrain.new_model_2d import newFeature
from EppflowLEAStereo.retrain.skip_model_3d import newMatching

import torch
import torch.nn as nn
import numpy as np
from eppcore import eppcore_inflation, eppcore_compression
import torch.nn.functional as F

class LEAStereo(nn.Module):
    def __init__(self, args):
        super(LEAStereo, self).__init__()

        network_path_fea, cell_arch_fea = np.load(args.net_arch_fea), np.load(args.cell_arch_fea)
        network_path_mat, cell_arch_mat = np.load(args.net_arch_mat), np.load(args.cell_arch_mat)
        print('Feature network path:{}\nMatching network path:{} \n'.format(network_path_fea, network_path_mat))

        network_arch_fea = network_layer_to_space(network_path_fea)
        network_arch_mat = network_layer_to_space(network_path_mat)

        self.feature = newFeature(network_arch_fea, cell_arch_fea, args=args)
        self.matching = newMatching(network_arch_mat, cell_arch_mat, args=args)
        self.args = args

        # Sample absolute depth
        linlogdedge = np.linspace(np.log(self.args.min_depth_pred), np.log(self.args.max_depth_pred), self.args.num_deges)
        linlogdedge = np.exp(linlogdedge)
        reld_edges = torch.from_numpy(linlogdedge).float().view([1, self.args.num_deges, 1, 1, 1, 1]).expand([-1, -1, self.args.inheight, self.args.inwidth, -1, -1])
        self.reld_edges = nn.Parameter(torch.from_numpy(linlogdedge).float().view([1, self.args.num_deges, 1, 1]), requires_grad=False)
        self.nedges = self.args.num_deges

        xx, yy = np.meshgrid(range(self.args.inwidth), range(self.args.inheight), indexing='xy')
        xx = torch.from_numpy(xx).float().view([1, 1, self.args.inheight, self.args.inwidth, 1, 1]).expand([-1, self.nedges, -1, -1, -1, -1])
        yy = torch.from_numpy(yy).float().view([1, 1, self.args.inheight, self.args.inwidth, 1, 1]).expand([-1, self.nedges, -1, -1, -1, -1])
        pts3d = torch.cat([xx * reld_edges, yy * reld_edges, reld_edges, torch.ones_like(xx)], axis=4)
        self.pts3d = nn.Parameter(pts3d.float().contiguous(), requires_grad=False)

        self.eppinflate = eppcore_inflation.apply
        self.eppcompress = eppcore_compression.apply

        self.disp = Disp()

    def resample_feature(self, feature1, feature2, instance, intrinsic, pose, pts3d):
        bz, featurc, featureh, featurew = feature1.shape
        sample_pts = self.get_samplecoords(instance, intrinsic, pose, pts3d, bz, featureh, featurew)

        feature2_ex = feature2.unsqueeze(1).expand([-1, self.nedges, -1, -1, -1]).contiguous().view([bz * self.nedges, featurc, featureh, featurew])
        sampled_feature2 = F.grid_sample(feature2_ex, sample_pts, mode='bilinear', align_corners=False).view([bz, self.nedges, featurc, featureh, featurew]).permute([0, 2, 1, 3, 4])
        feature_volume = torch.cat([feature1.unsqueeze(2).expand([-1, -1, self.nedges, -1, -1]), sampled_feature2], dim=1)
        return feature_volume

    def get_samplecoords(self, instance, intrinsic, pose, pts3d, bz, featureh, featurew):
        if pts3d is None:
            pts3d = self.pts3d
        pts3d = pts3d.expand([bz, -1, -1, -1, -1, -1])

        intrinsic_ex = intrinsic.view(bz, 1, 4, 4).expand([-1, self.args.maxinsnum, -1, -1])
        P = intrinsic_ex @ pose @ torch.inverse(intrinsic_ex)
        P_inf = self.eppinflate(instance, P)

        pts3d_proj = P_inf.unsqueeze(1).expand([-1, self.nedges, -1, -1, -1, -1]) @ pts3d

        px, py, d, ones = torch.split(pts3d_proj, 1, dim=4)

        sign = d.sign()
        sign[sign == 0] = 1
        d = torch.clamp(torch.abs(d), min=1e-20) * sign

        sample_px = (px / d).squeeze(-1).squeeze(-1) / 3
        sample_px = F.interpolate(sample_px, [featureh, featurew], mode='nearest')
        sample_px = (sample_px / featurew - 0.5) * 2

        sample_py = (py / d).squeeze(-1).squeeze(-1) / 3
        sample_py = F.interpolate(sample_py, [featureh, featurew], mode='nearest')
        sample_py = (sample_py / featureh - 0.5) * 2

        sample_pts = torch.stack([sample_px, sample_py], dim=-1).view(bz * self.nedges, featureh, featurew, 2)
        return sample_pts

    def forward(self, x, y, insmap, intrinsic, pose, pts3d=None):
        bz, _, h, w = x.shape

        x = self.feature(x)
        y = self.feature(y)

        cost = self.resample_feature(x, y, insmap, intrinsic, pose, pts3d)
        cost = self.matching(cost)
        disp = self.disp(cost, self.reld_edges)

        return disp

    def get_params(self):
        back_bn_params, back_no_bn_params = self.encoder.get_params()
        tune_wd_params = list(self.aspp.parameters()) + list(self.decoder.parameters()) + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params

    def validate_sample(self, image1, image2, depthmap, flowmap, instance, intrinsic, pose):
        ratio = 3
        bz, _, h, w = image1.shape

        featureh = int(h / ratio)
        featurew = int(w / ratio)

        pts3d = self.pts3d
        pts3d = pts3d.expand([bz, -1, -1, -1, -1, -1])

        sample_pts = self.get_samplecoords(instance, intrinsic, pose, pts3d, bz, featureh, featurew)
        sample_pts = sample_pts.view([bz, self.nedges, featureh, featurew, 2])
        sample_ptsx = (sample_pts[0, :, :, :, 0].cpu().detach().numpy() + 1) / 2 * w
        sample_ptsy = (sample_pts[0, :, :, :, 1].cpu().detach().numpy() + 1) / 2 * h

        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        selector = (instance[0].squeeze().detach().cpu().numpy() >= 1) * (np.mod(xx, ratio) == 0) * (np.mod(yy, ratio) == 0)
        xxf = xx[selector]
        yyf = yy[selector]

        rndidx = np.random.randint(0, xxf.shape[0], 1).item()
        rndx = int(xxf[rndidx] / ratio)
        rndy = int(yyf[rndidx] / ratio)
        rndd = depthmap[0, 0, int(rndy * ratio), int(rndx * ratio)].item()
        rndins = instance[0, 0, int(rndy * ratio), int(rndx * ratio)].item()
        rndsamplex = sample_ptsx[:, rndy, rndx]
        rndsampley = sample_ptsy[:, rndy, rndx]

        rndpts3d = np.array([[rndx * ratio * rndd, rndy * ratio * rndd, rndd, 1]]).T
        intrinsicnp = intrinsic[0].detach().cpu().numpy()
        Pnp = pose[0, rndins, :, :].detach().cpu().numpy()
        rndpts3d_p = intrinsicnp @ Pnp @ np.linalg.inv(intrinsicnp) @ rndpts3d
        rndpts3d_p[0, 0] = rndpts3d_p[0, 0] / rndpts3d_p[2, 0]
        rndpts3d_p[1, 0] = rndpts3d_p[1, 0] / rndpts3d_p[2, 0]

        flowmapnp = flowmap[0].detach().cpu().numpy()
        gtx = rndx * ratio + flowmapnp[0, int(rndy * ratio), int(rndx * ratio)]
        gty = rndy * ratio + flowmapnp[1, int(rndy * ratio), int(rndx * ratio)]

        import matplotlib.pyplot as plt
        image1np = image1[0].cpu().permute([1, 2, 0]).numpy().astype(np.uint8)
        image2np = image2[0].cpu().permute([1, 2, 0]).numpy().astype(np.uint8)

        fig = plt.figure(figsize=(16, 9))
        fig.add_subplot(2, 1, 1)
        plt.scatter(rndx * ratio, rndy * ratio, 10, 'r')
        plt.imshow(image1np)

        fig.add_subplot(2, 1, 2)
        plt.scatter(rndsamplex, rndsampley, 2, 'r')
        plt.scatter(rndpts3d_p[0], rndpts3d_p[1], 10, 'g')
        plt.imshow(image2np)

        plt.show()