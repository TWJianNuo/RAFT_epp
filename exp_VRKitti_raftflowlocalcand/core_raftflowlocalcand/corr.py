import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler, coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        self.corr = corr.reshape(batch*h1*w1, dim, h2, w2)

    def __call__(self, coords, flowmag, ratio):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        corr = self.corr
        dx = torch.linspace(-r, r, 2*r+1)
        dy = torch.linspace(-r, r, 2*r+1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

        centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2)
        delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2).expand([batch*h1*w1, -1, -1, -1]) * flowmag.view(batch*h1*w1, 1, 1, 1).expand([-1, 2*r+1, 2*r+1, 2]) * ratio
        coords_lvl = centroid_lvl + delta_lvl

        corr = bilinear_sampler(corr, coords_lvl)
        corr = corr.view(batch, h1, w1, -1)

        corr = corr.permute(0, 3, 1, 2).contiguous().float()
        coords_lvl = coords_lvl.view(batch, h1, w1, 2*r+1, 2*r+1, 2)
        delta_lvl = delta_lvl.view(batch, h1, w1, (2*r+1) * (2*r+1), 2)
        return corr, coords_lvl, delta_lvl

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())