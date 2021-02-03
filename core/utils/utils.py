import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
import PIL.Image as Image
import matplotlib.pyplot as plt

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def tensor2disp(tensor, vmax=0.18, percentile=None, viewind=0):
    cm = plt.get_cmap('magma')
    tnp = tensor[viewind, 0, :, :].detach().cpu().numpy()
    if percentile is not None:
        vmax = np.percentile(tnp, percentile)
    tnp = tnp / vmax
    tnp = (cm(tnp) * 255).astype(np.uint8)
    return Image.fromarray(tnp[:, :, 0:3])

def tensor2grad(gradtensor, percentile=95, pos_bar=0, neg_bar=0, viewind=0):
    cm = plt.get_cmap('bwr')
    gradnumpy = gradtensor.detach().cpu().numpy()[viewind, 0, :, :]

    selector_pos = gradnumpy > 0
    if np.sum(selector_pos) > 1:
        if pos_bar <= 0:
            pos_bar = np.percentile(gradnumpy[selector_pos], percentile)
        gradnumpy[selector_pos] = gradnumpy[selector_pos] / pos_bar / 2

    selector_neg = gradnumpy < 0
    if np.sum(selector_neg) > 1:
        if neg_bar >= 0:
            neg_bar = -np.percentile(-gradnumpy[selector_neg], percentile)
        gradnumpy[selector_neg] = -gradnumpy[selector_neg] / neg_bar / 2

    disp_grad_numpy = gradnumpy + 0.5
    colorMap = cm(disp_grad_numpy)[:,:,0:3]
    return Image.fromarray((colorMap * 255).astype(np.uint8))