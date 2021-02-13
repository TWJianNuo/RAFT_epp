import torch
import eppcoreops
import torch.nn as nn
import numpy as np
import time

def debug_epp_inflation(instance, infdst, infsrc, bz):
    rndbz = int(np.random.randint(0, bz, 1))
    rndins = int(np.random.randint(0, instance[rndbz].max().item(), 1))

    dstM = infdst[rndbz, instance[rndbz, 0] == rndins, :, :]
    srcM = infsrc[rndbz, rndins]

    absdiff = torch.mean(torch.abs(dstM - srcM))
    print("absolute difference is: %f" % (absdiff))

def debug_epp_compression(instance, compdst, compsrc, bz):
    # rndbz = int(np.random.randint(0, bz, 1))
    # rndins = int(np.random.randint(0, instance[rndbz, 0].max().item(), 1))
    rndbz = 0
    rndins = 0

    scatterM = compsrc[rndbz, instance[rndbz, 0] == rndins, :, :]
    conpressedM = torch.sum(scatterM, dim=0)

    absdiff = torch.max(torch.abs(conpressedM - compdst[rndbz, rndins]))
    print("absolute difference is: %f" % (absdiff))

class eppcore_inflation(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __int__(self):
        super(eppcore_inflation, self).__init__()

    @staticmethod
    def forward(ctx, instance, infsrc):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # ctx.save_for_backward(pred_log, semantics, mask, variance, depthin, depthout, summedconfidence, sdirintexpvariance, sdirintweighteddepth, sdirlateralre)
        # ctx.h = h
        # ctx.w = w
        # ctx.bz = bz
        # ctx.clipvariance = clipvariance
        # ctx.maxrange = maxrange
        assert instance.dtype == torch.int32 and infsrc.dtype == torch.float32, print("epp inflation input data type error")
        assert instance.ndim == 4 and infsrc.ndim == 4, print("epp inflation input shape error")

        bz, _, h, w = instance.shape
        _, _, infh, infw = infsrc.shape
        device = infsrc.device
        infdst = torch.zeros([bz, h, w, infh, infw], dtype=torch.float32, device=device)
        eppcoreops.epp_inflation(instance, infdst, infsrc, h, w, bz, infh, infw)
        # debug_epp_inflation(instance, infdst, infsrc, bz)

        return infdst

class eppcore_compression(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __int__(self):
        super(eppcore_compression, self).__init__()

    @staticmethod
    def forward(ctx, instance, compsrc, maxinsnum):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # ctx.save_for_backward(pred_log, semantics, mask, variance, depthin, depthout, summedconfidence, sdirintexpvariance, sdirintweighteddepth, sdirlateralre)
        # ctx.h = h
        # ctx.w = w
        # ctx.bz = bz
        # ctx.clipvariance = clipvariance
        # ctx.maxrange = maxrange
        assert instance.dtype == torch.int32 and compsrc.dtype == torch.float32, print("epp compression input data type error")
        assert instance.ndim == 4 and compsrc.ndim == 5, print("epp compression input shape error")

        bz, _, h, w = instance.shape
        _, _, _, comph, compw = compsrc.shape
        device = compsrc.device
        compdst = torch.zeros([bz, maxinsnum, comph, compw], dtype=torch.float32, device=device)

        eppcoreops.epp_compression(instance, compdst, compsrc * 1.1, h, w, bz, comph, compw)
        print(compdst[0,0,0,0], compdst[1,0,0,0])
        debug_epp_compression(instance, compdst, compsrc * 1.1, bz)

        return compdst

class EPPCore(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, height, width, bz, itnum=100, lr=0.1, lap=1e-2):
        super(EPPCore, self).__init__()
        self.height = height
        self.width = width
        self.bz = bz
        self.maxinsnum = 200

        self.itnum = itnum
        self.lr = lr
        self.lap = lap

        self.init_pts()
        self.init_t2TM()
        self.init_ang2RM()
        self.init_t_ang()

        self.init_deriv_t()
        self.init_ang_derivM()

        self.eppinflate = eppcore_inflation.apply
        self.eppcompress = eppcore_compression.apply

    def init_t_ang(self):
        t_init = torch.zeros([self.bz, self.maxinsnum, 3, 1])
        t_init[:, :, 2, :] = -0.95
        ang_init = torch.zeros([self.bz, self.maxinsnum, 3, 1])
        self.t_init = nn.Parameter(t_init, requires_grad=False)
        self.ang_init = nn.Parameter(ang_init, requires_grad=False)

    def init_pts(self):
        xx, yy = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        pts = np.stack([xx, yy, np.ones_like(xx)], axis=2)
        pts = torch.from_numpy(pts).float().unsqueeze(3).unsqueeze(0).expand([self.bz, -1, -1, -1, -1])
        self.pts = nn.Parameter(pts, requires_grad=False)

    def init_t2TM(self):
        t2TM = torch.zeros([3, 3, 3])
        t2TM[0, 1, 2] = -1
        t2TM[0, 2, 1] = 1
        t2TM[1, 0, 2] = 1
        t2TM[1, 2, 0] = -1
        t2TM[2, 0, 1] = -1
        t2TM[2, 1, 0] = 1
        t2TM = t2TM.view([1, 1, 3, 3, 1, 3]).expand([self.bz, self.maxinsnum, -1, -1, -1, -1])
        self.t2TM = torch.nn.Parameter(t2TM, requires_grad=False)

    def init_deriv_t(self):
        T0 = torch.zeros([3, 3])
        T1 = torch.zeros([3, 3])
        T2 = torch.zeros([3, 3])

        T0[1, 2] = -1
        T0[2, 1] = 1

        T1[0, 2] = 1
        T1[2, 0] = -1

        T2[0, 1] = -1
        T2[1, 0] = 1

        T0 = T0.view([1, 1, 1, 3, 3]).expand([self.bz, self.height, self.width, -1, -1])
        T1 = T1.view([1, 1, 1, 3, 3]).expand([self.bz, self.height, self.width, -1, -1])
        T2 = T2.view([1, 1, 1, 3, 3]).expand([self.bz, self.height, self.width, -1, -1])

        self.T0_inf = torch.nn.Parameter(T0, requires_grad=False)
        self.T1_inf = torch.nn.Parameter(T1, requires_grad=False)
        self.T2_inf = torch.nn.Parameter(T2, requires_grad=False)

        return T0, T1, T2

    def t2T(self, t):
        te = t.view([self.bz, self.maxinsnum, 1, 1, 3, 1]).expand([-1, -1, 3, 3, -1, -1])
        T = (self.t2TM @ te).squeeze(-1).squeeze(-1)

        # self.debug_t2T(t, T)

        return T

    def debug_t2T(self, t, T):
        rndbz = int(np.random.randint(0, self.bz))
        rndins = int(np.random.randint(0, 3))

        t_spl = t[rndbz, rndins]
        T_val = torch.zeros([3, 3], device=t.device, dtype=torch.float)
        T_val[0, 1] = -t_spl[2, 0]
        T_val[0, 2] = t_spl[1, 0]
        T_val[1, 0] = t_spl[2, 0]
        T_val[1, 2] = -t_spl[0, 0]
        T_val[2, 0] = -t_spl[1, 0]
        T_val[2, 1] = t_spl[0, 0]

        diff = (T_val - T[rndbz, rndins]).abs().mean()
        print("Numerical diff is %f" % diff)

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

        ang2RMx = ang2RMx.view([1, 1, 3, 3, 1, 6]).expand([self.bz, self.maxinsnum, -1, -1, -1, -1])
        self.ang2RMx = torch.nn.Parameter(ang2RMx, requires_grad=False)
        ang2RMy = ang2RMy.view([1, 1, 3, 3, 1, 6]).expand([self.bz, self.maxinsnum, -1, -1, -1, -1])
        self.ang2RMy = torch.nn.Parameter(ang2RMy, requires_grad=False)
        ang2RMz = ang2RMz.view([1, 1, 3, 3, 1, 6]).expand([self.bz, self.maxinsnum, -1, -1, -1, -1])
        self.ang2RMz = torch.nn.Parameter(ang2RMz, requires_grad=False)

        rotxcomp = torch.zeros([3, 3])
        rotxcomp[0, 0] = 1
        rotxcomp = rotxcomp.view([1, 1, 3, 3]).expand([self.bz, self.maxinsnum, -1, -1])
        self.rotxcomp = torch.nn.Parameter(rotxcomp, requires_grad=False)

        rotycomp = torch.zeros([3, 3])
        rotycomp[1, 1] = 1
        rotycomp = rotycomp.view([1, 1, 3, 3]).expand([self.bz, self.maxinsnum, -1, -1])
        self.rotycomp = torch.nn.Parameter(rotycomp, requires_grad=False)

        rotzcomp = torch.zeros([3, 3])
        rotzcomp[2, 2] = 1
        rotzcomp = rotzcomp.view([1, 1, 3, 3]).expand([self.bz, self.maxinsnum, -1, -1])
        self.rotzcomp = torch.nn.Parameter(rotzcomp, requires_grad=False)

    def init_ang_derivM(self):
        ang2RM_derivx = torch.zeros([3, 3, 6])
        ang2RM_derivy = torch.zeros([3, 3, 6])
        ang2RM_derivz = torch.zeros([3, 3, 6])

        ang2RM_derivx[1, 1, 0 + 3] = -1
        ang2RM_derivx[1, 2, 0] = -1
        ang2RM_derivx[2, 1, 0] = 1
        ang2RM_derivx[2, 2, 0 + 3] = -1

        ang2RM_derivy[0, 0, 1 + 3] = -1
        ang2RM_derivy[0, 2, 1] = 1
        ang2RM_derivy[2, 0, 1] = -1
        ang2RM_derivy[2, 2, 1 + 3] = -1

        ang2RM_derivz[0, 0, 2 + 3] = -1
        ang2RM_derivz[0, 1, 2] = -1
        ang2RM_derivz[1, 0, 2] = 1
        ang2RM_derivz[1, 1, 2 + 3] = -1

        ang2RM_derivx = ang2RM_derivx.view([1, 1, 3, 3, 1, 6]).expand([self.bz, self.maxinsnum, -1, -1, -1, -1])
        self.ang2RM_derivx = torch.nn.Parameter(ang2RM_derivx, requires_grad=False)
        ang2RM_derivy = ang2RM_derivy.view([1, 1, 3, 3, 1, 6]).expand([self.bz, self.maxinsnum, -1, -1, -1, -1])
        self.ang2RM_derivy = torch.nn.Parameter(ang2RM_derivy, requires_grad=False)
        ang2RM_derivz = ang2RM_derivz.view([1, 1, 3, 3, 1, 6]).expand([self.bz, self.maxinsnum, -1, -1, -1, -1])
        self.ang2RM_derivz = torch.nn.Parameter(ang2RM_derivz, requires_grad=False)

    def debug_ang2R(self, angs, rot, rotxd_val, rotyd_val, rotzd_val):
        rndbz = int(np.random.randint(0, self.bz, 1))
        rndins = int(np.random.randint(0, self.maxinsnum, 1))

        angsspl = angs[rndbz, rndins, :, 0]
        rotspl = rot[rndbz, rndins]

        rotx = torch.eye(3, device=angs.device)
        roty = torch.eye(3, device=angs.device)
        rotz = torch.eye(3, device=angs.device)

        rotx[1, 1] = torch.cos(angsspl[0])
        rotx[1, 2] = -torch.sin(angsspl[0])
        rotx[2, 1] = torch.sin(angsspl[0])
        rotx[2, 2] = torch.cos(angsspl[0])

        roty[0, 0] = torch.cos(angsspl[1])
        roty[0, 2] = torch.sin(angsspl[1])
        roty[2, 0] = -torch.sin(angsspl[1])
        roty[2, 2] = torch.cos(angsspl[1])

        rotz[0, 0] = torch.cos(angsspl[2])
        rotz[0, 1] = -torch.sin(angsspl[2])
        rotz[1, 0] = torch.sin(angsspl[2])
        rotz[1, 1] = torch.cos(angsspl[2])

        rotref = rotz @ (roty @ rotx)

        rotxd = torch.zeros([3, 3], device=angs.device)
        rotyd = torch.zeros([3, 3], device=angs.device)
        rotzd = torch.zeros([3, 3], device=angs.device)

        rotxd[1, 1] = -torch.sin(angsspl[0])
        rotxd[1, 2] = -torch.cos(angsspl[0])
        rotxd[2, 1] = torch.cos(angsspl[0])
        rotxd[2, 2] = -torch.sin(angsspl[0])

        rotyd[0, 0] = -torch.sin(angsspl[1])
        rotyd[0, 2] = torch.cos(angsspl[1])
        rotyd[2, 0] = -torch.cos(angsspl[1])
        rotyd[2, 2] = -torch.sin(angsspl[1])

        rotzd[0, 0] = -torch.sin(angsspl[2])
        rotzd[0, 1] = -torch.cos(angsspl[2])
        rotzd[1, 0] = torch.cos(angsspl[2])
        rotzd[1, 1] = -torch.sin(angsspl[2])

        rotxd = rotz @ roty @ rotxd
        rotyd = rotz @ rotyd @ rotx
        rotzd = rotzd @ roty @ rotx

        diff = (rotspl - rotref).abs().mean()
        diffx = (rotxd - rotxd_val[rndbz, rndins]).abs().mean()
        diffy = (rotyd - rotyd_val[rndbz, rndins]).abs().mean()
        diffz = (rotzd - rotzd_val[rndbz, rndins]).abs().mean()
        print("Difference with referred rotationM is: %f, diffx is: %f, diffy is: %f, diffz is: %f" % (diff, diffx, diffy, diffz))

    def ang2R(self, angs, requires_deriv=False):
        """Convert an axisangle rotation into a 4x4 transformation matrix
        (adapted from https://github.com/Wallacoloo/printipi)
        Input 'vec' has to be Bx1x3
        """
        cos_sin = torch.cat([torch.cos(angs), torch.sin(angs)], dim=2).view([self.bz, self.maxinsnum, 1, 1, 6, 1]).expand([-1, -1, 3, 3, -1, -1])
        rotx = (self.ang2RMx @ cos_sin).squeeze(-1).squeeze(-1) + self.rotxcomp
        roty = (self.ang2RMy @ cos_sin).squeeze(-1).squeeze(-1) + self.rotycomp
        rotz = (self.ang2RMz @ cos_sin).squeeze(-1).squeeze(-1) + self.rotzcomp
        rot = rotz @ roty @ rotx

        if requires_deriv:
            rotxdt = (self.ang2RM_derivx @ cos_sin).squeeze(-1).squeeze(-1)
            rotydt = (self.ang2RM_derivy @ cos_sin).squeeze(-1).squeeze(-1)
            rotzdt = (self.ang2RM_derivz @ cos_sin).squeeze(-1).squeeze(-1)

            rotxd = rotz @ roty @ rotxdt
            rotyd = rotz @ rotydt @ rotx
            rotzd = rotzdt @ roty @ rotx

            # self.debug_ang2R(angs, rot, rotxd, rotyd, rotzd)
            return rot, rotxd, rotyd, rotzd
        else:
            return rot

    def flowmap2pts(self, flowmap):
        pts_inc = torch.cat([flowmap.permute([0, 2, 3, 1]).unsqueeze(-1), torch.zeros([self.bz, self.height, self.width, 1, 1], dtype=torch.float, device=flowmap.device)], dim=-2)
        pts1 = self.pts
        pts2 = self.pts + pts_inc

        return pts1, pts2

    def flow2epp(self, insmap, flowmap, intrinsic, t_init=None, ang_init=None):
        if t_init is None:
            t_init = self.t_init
        if ang_init is None:
            ang_init = self.ang_init

        # inscount = self.eppcompress(insmap, (insmap > -1).squeeze(1).unsqueeze(-1).unsqueeze(-1).float(), self.maxinsnum)
        testm = (insmap > -1).squeeze(1).unsqueeze(-1).unsqueeze(-1).float()
        # testm = torch.ones_like(testm)
        # testm = testm + (torch.rand_like(testm) * 100).round()
        # testm = torch.round(testm * 1e0) / 1e0
        inscount = self.eppcompress(insmap, testm, self.maxinsnum)

        intrinsic_inv = torch.inverse(intrinsic)
        intrinsic_inv_ex = intrinsic_inv.unsqueeze(1).expand([-1, self.maxinsnum, -1, -1])
        intrinsic_inv_inf = intrinsic_inv.view([self.bz, 1, 1, 3, 3]).expand([-1, self.height, self.width, -1, -1])

        pts1, pts2 = self.flowmap2pts(flowmap)
        ptsl = torch.transpose(torch.transpose(pts2, dim0=3, dim1=4) @ torch.transpose(intrinsic_inv_inf, dim0=3, dim1=4), dim0=3, dim1=4)
        ptsr = intrinsic_inv_inf @ pts1
        derivM = ptsl @ torch.transpose(ptsr, dim0=3, dim1=4)

        t = t_init
        ang = ang_init
        for k in range(self.itnum):
            T = self.t2T(t)
            R, rotxd, rotyd, rotzd = self.ang2R(torch.rand_like(ang), requires_deriv=True)

            T_inf = self.eppinflate(insmap, T)
            R_inf = self.eppinflate(insmap, R)
            rotxd_inf = self.eppinflate(insmap, rotxd)
            rotyd_inf = self.eppinflate(insmap, rotyd)
            rotzd_inf = self.eppinflate(insmap, rotzd)

            r = (torch.transpose(ptsl, dim0=3, dim1=4) @ T_inf @ R_inf @ ptsr)

            tnorm = torch.norm(t, dim=[2, 3], keepdim=True)
            J_t0_bias = 2 * self.lap * (tnorm - 1) / tnorm * t[:, :, 0:1, :]
            J_t1_bias = 2 * self.lap * (tnorm - 1) / tnorm * t[:, :, 1:2, :]
            J_t2_bias = 2 * self.lap * (tnorm - 1) / tnorm * t[:, :, 2:3, :]
            J_t0_bias_inf = self.eppinflate(insmap, J_t0_bias)
            J_t1_bias_inf = self.eppinflate(insmap, J_t1_bias)
            J_t2_bias_inf = self.eppinflate(insmap, J_t2_bias)

            J_t0 = torch.sum(derivM * (self.T0_inf @ R_inf), dim=[3, 4], keepdim=True) * 2 * r + J_t0_bias_inf
            J_t1 = torch.sum(derivM * (self.T1_inf @ R_inf), dim=[3, 4], keepdim=True) * 2 * r + J_t1_bias_inf
            J_t2 = torch.sum(derivM * (self.T2_inf @ R_inf), dim=[3, 4], keepdim=True) * 2 * r + J_t2_bias_inf

            J_ang0 = torch.sum(derivM * (T_inf @ rotxd_inf), dim=[3, 4], keepdim=True) * 2 * r
            J_ang1 = torch.sum(derivM * (T_inf @ rotyd_inf), dim=[3, 4], keepdim=True) * 2 * r
            J_ang2 = torch.sum(derivM * (T_inf @ rotzd_inf), dim=[3, 4], keepdim=True) * 2 * r

            JacobM = torch.cat([J_ang0, J_ang1, J_ang2, J_t0, J_t1, J_t2], dim=-2)
            MM = JacobM @ torch.transpose(JacobM, dim1=3, dim0=4)
            self.eppcompress(insmap, MM, self.maxinsnum)
            a = 1

        return

    def get_eppcons(self, insmap, flowmap, intrinsic, poses):
        t = poses[:, :, 0:3, 3:4]
        t_norm = t / (torch.norm(t, dim=[2, 3], keepdim=True) + 1e-12)
        T = self.t2T(t_norm)
        R = poses[:, :, 0:3, 0:3]

        intrinsic_inv = torch.inverse(intrinsic)
        intrinsic_inv_ex = intrinsic_inv.unsqueeze(1).expand([-1, self.maxinsnum, -1, -1])
        E = intrinsic_inv_ex.transpose(dim0=2, dim1=3) @ T @ R @ intrinsic_inv_ex
        E_inf = self.eppinflate(insmap, E)
        pts1, pts2 = self.flowmap2pts(flowmap)

        constrain_inf = torch.sum(torch.transpose(pts2, dim0=-2, dim1=-1) @ E_inf * torch.transpose(pts1, dim0=-2, dim1=-1), dim=[-1, -2]).unsqueeze(1)
        return constrain_inf



