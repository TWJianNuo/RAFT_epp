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
    rndbz = int(np.random.randint(0, bz, 1))
    rndins = int(np.random.randint(0, instance[rndbz, 0].max().item(), 1))

    scatterM = compsrc[rndbz, instance[rndbz, 0] == rndins, :, :]
    conpressedM = torch.sum(scatterM, dim=0)

    absdiff = torch.max(torch.abs(conpressedM - compdst[rndbz, rndins]))
    print("absolute difference is: %f" % (absdiff))

def debug_epp_batchselection():
    bz = 2
    maxinsnum = 200
    possiblenum = 10
    srch = 3
    srcw = 3

    selsrc = torch.rand([bz, maxinsnum, possiblenum, srch, srcw], dtype=torch.float, device=torch.device("cuda"), requires_grad=True)
    batchidx = torch.randint(0, possiblenum, [bz, maxinsnum], dtype=torch.int, device=torch.device("cuda"))
    eppsel = eppcore_batchsel.apply
    seldst = eppsel(batchidx, selsrc)

    rndbz = np.random.randint(0, bz)
    rndins = np.random.randint(0, maxinsnum)
    rndh = np.random.randint(0, srch)
    rndw = np.random.randint(0, srcw)
    delta = 1e-1

    diff = (selsrc[rndbz, rndins, batchidx[rndbz, rndins], :, :] - seldst[rndbz, rndins]).abs().max()

    def lossfunc(input):
        return (input.abs()).sum()

    lossfunc(seldst).backward()
    selsrc_grad = selsrc.grad

    selsrc_pos = torch.clone(selsrc)
    selsrc_neg = torch.clone(selsrc)
    selsrc_pos[rndbz, rndins, batchidx[rndbz, rndins], rndh, rndw] += delta
    selsrc_neg[rndbz, rndins, batchidx[rndbz, rndins], rndh, rndw] -= delta

    numgrad = (lossfunc(eppsel(batchidx, selsrc_pos)) - lossfunc(eppsel(batchidx, selsrc_neg))) / 2 / delta
    theograd = selsrc_grad[rndbz, rndins, batchidx[rndbz, rndins], rndh, rndw]

    isones = (selsrc_grad[rndbz, rndins, batchidx[rndbz, rndins], :, :] - torch.ones([srch, srcw], device=torch.device("cuda"))).abs().max()

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
        assert instance.dtype == torch.int32 and infsrc.dtype == torch.float32, print("epp inflation input data type error")
        assert instance.ndim == 4 and infsrc.ndim == 4, print("epp inflation input shape error")

        bz, _, h, w = instance.shape
        _, maxinsnum, infh, infw = infsrc.shape
        infdst = torch.zeros([bz, h, w, infh, infw], dtype=torch.float32, device=infsrc.device)
        eppcoreops.epp_inflation(instance, infdst, infsrc, h, w, bz, infh, infw)
        # debug_epp_inflation(instance, infdst, infsrc, bz)

        ctx.save_for_backward(instance)
        ctx.h = h
        ctx.w = w
        ctx.bz = bz
        ctx.maxinsnum = maxinsnum
        ctx.infh = infh
        ctx.infw = infw
        return infdst

    @staticmethod
    def backward(ctx, grad_infsrc):
        h = ctx.h
        w = ctx.w
        bz = ctx.bz
        maxinsnum = ctx.maxinsnum
        comph = ctx.infh
        compw = ctx.infw

        instance, = ctx.saved_tensors

        grad_infsrc = grad_infsrc.contiguous()

        compdst = torch.zeros([bz, maxinsnum, comph, compw], dtype=torch.double, device=grad_infsrc.device)
        eppcoreops.epp_compression(instance, compdst, grad_infsrc, h, w, bz, comph, compw)
        compdst = compdst.float()

        return None, compdst

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
        assert instance.dtype == torch.int32 and compsrc.dtype == torch.float32, print("epp compression input data type error")
        assert instance.ndim == 4 and compsrc.ndim == 5, print("epp compression input shape error")

        bz, _, h, w = instance.shape
        _, _, _, comph, compw = compsrc.shape
        compdst = torch.zeros([bz, maxinsnum, comph, compw], dtype=torch.double, device=compsrc.device)

        eppcoreops.epp_compression(instance, compdst, compsrc, h, w, bz, comph, compw)
        compdst = compdst.float()
        # debug_epp_compression(instance, compdst, compsrc, bz)

        ctx.save_for_backward(instance)
        ctx.h = h
        ctx.w = w
        ctx.bz = bz
        ctx.comph = comph
        ctx.compw = compw
        return compdst

    @staticmethod
    def backward(ctx, grad_compdst):
        h = ctx.h
        w = ctx.w
        bz = ctx.bz
        infh = ctx.comph
        infw = ctx.compw

        instance, = ctx.saved_tensors

        grad_compdst = grad_compdst.contiguous()

        infdst = torch.zeros([bz, h, w, infh, infw], dtype=torch.float32, device=grad_compdst.device)
        eppcoreops.epp_inflation(instance, infdst, grad_compdst, h, w, bz, infh, infw)

        return None, infdst, None

class eppcore_batchsel(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __int__(self):
        super(eppcore_batchsel, self).__init__()

    @staticmethod
    def forward(ctx, batchidx, selsrc):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        bz, insnum, candidatenum, srch, srcw = selsrc.shape
        ck1, ck2 = batchidx.shape

        assert bz == ck1 and insnum == ck2, print("index shound be in correspondence with select source")
        assert batchidx.dtype == torch.int and selsrc.dtype == torch.float32, print("epp compression input data type error")
        assert batchidx.ndim == 2 and selsrc.ndim == 5, print("epp compression input shape error")

        device = selsrc.device
        seldst = torch.zeros([bz, insnum, srch, srcw], dtype=torch.float32, device=device)

        eppcoreops.epp_batchselection(batchidx, seldst, selsrc, insnum, bz, srch, srcw)

        ctx.save_for_backward(batchidx)
        ctx.bz = bz
        ctx.insnum = insnum
        ctx.candidatenum = candidatenum
        ctx.srch = srch
        ctx.srcw = srcw
        return seldst

    @staticmethod
    def backward(ctx, grad_seldst):
        bz = ctx.bz
        insnum = ctx.insnum
        candidatenum = ctx.candidatenum
        srch = ctx.srch
        srcw = ctx.srcw

        batchidx, = ctx.saved_tensors

        grad_seldst = grad_seldst.contiguous()

        device = grad_seldst.device
        grad_selsrc = torch.zeros([bz, insnum, candidatenum, srch, srcw], dtype=torch.float32, device=device)
        eppcoreops.epp_batchselection_backward(batchidx, grad_selsrc, grad_seldst, insnum, bz, srch, srcw)

        return None, grad_selsrc

class eppcore_selupdate(nn.Module):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __int__(self):
        super(eppcore_compression, self).__init__()

    def forward(self, updateMl, updateMr, insnum, r_current, r_last, terminateflag, idtpadding):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        assert insnum.dtype == torch.float32 and updateMl.dtype == torch.float32 and updateMr.dtype == torch.float32, print("epp compression input data type error")
        assert insnum.ndim == 4 and updateMl.ndim == 4 and updateMr.ndim == 4, print("epp compression input shape error")

        bz, maxinsnum, _, _ = updateMl.shape

        updateMl_padded = updateMl + (insnum == 0).float().expand([-1, -1, 6, 6]) * idtpadding
        updateMl_padded_inv = torch.inverse(updateMl_padded)
        updateM = updateMl_padded_inv @ updateMr

        conditionck, _ = torch.max((updateMl_padded @ updateMl_padded_inv - idtpadding).abs().view([bz, maxinsnum, -1]), dim=-1, keepdim=True)
        conditionck = conditionck.unsqueeze(-1) < 0.1

        terminateflag = terminateflag * (r_current < r_last)
        terminateflag = terminateflag * conditionck
        if torch.sum(terminateflag) == 0:
            return None, None, None
        else:
            updateM = updateM * terminateflag.float().expand([-1, -1, 6, -1])
            updateang, updatel = torch.split(updateM, 3, dim=2)
            return updateang, updatel, terminateflag

class EPPCore(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, height, width, bz, itnum=100, lr=0.2, lap=1e-2, maxinsnum=200):
        super(EPPCore, self).__init__()
        self.height = height
        self.width = width
        self.bz = bz
        self.maxinsnum = maxinsnum

        self.itnum = itnum
        self.lr = lr
        self.lap = lap

        self.init_pts()
        self.init_t2TM()
        self.init_ang2RM()
        self.init_t_ang()

        self.init_deriv_t()
        self.init_ang_derivM()

        self.init_idtpadding()

        self.init_negativesign()

        self.eppinflate = eppcore_inflation.apply
        self.eppcompress = eppcore_compression.apply
        self.eppsel = eppcore_batchsel.apply
        self.eppupdate = eppcore_selupdate()

    def init_negativesign(self):
        negativesign = -torch.ones([3, 3], dtype=torch.float32)
        negativesign = negativesign.view([1, 1, 3, 3]).expand([self.bz, self.maxinsnum, -1, -1])
        self.negativesign = nn.Parameter(negativesign, requires_grad=False)

    def init_idtpadding(self):
        idtpadding = torch.eye(6)
        idtpadding = idtpadding.view([1, 1, 6, 6]).expand([self.bz, self.maxinsnum, -1, -1])
        self.idtpadding = nn.Parameter(idtpadding, requires_grad=False)

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

    def R2ang(self, R):
        # This is not an efficient implementation
        sy = torch.sqrt(R[:, :, 0, 0] * R[:, :, 0, 0] + R[:, :, 1, 0] * R[:, :, 1, 0])
        ang0 = torch.atan2(R[:, :, 2, 1], R[:, :, 2, 2])
        ang1 = torch.atan2(-R[:, :, 2, 0], sy)
        ang2 = torch.atan2(R[:, :, 1, 0], R[:, :, 0, 0])
        ang = torch.stack([ang0, ang1, ang2], dim=2).unsqueeze(-1)
        return ang

    def flowmap2pts(self, flowmap):
        pts_inc = torch.cat([flowmap.permute([0, 2, 3, 1]).unsqueeze(-1), torch.zeros([self.bz, self.height, self.width, 1, 1], dtype=torch.float, device=flowmap.device)], dim=-2)
        pts1 = self.pts
        pts2 = self.pts + pts_inc

        return pts1, pts2

    def extract_tR_combs(self, t, ang):
        E = self.t2T(t) @ self.ang2R(ang)
        e1, e2, e3 = torch.split(E, 1, dim=3)

        w1 = torch.cross(t, e1, dim=2)
        w2 = torch.cross(t, e2, dim=2)
        w3 = torch.cross(t, e3, dim=2)

        r11 = w1 + torch.cross(w2, w3, dim=2)
        r12 = w2 + torch.cross(w3, w1, dim=2)
        r13 = w3 + torch.cross(w1, w2, dim=2)
        recoverR1 = torch.cat([r11, r12, r13], dim=3)

        r21 = w1 + torch.cross(w3, w2, dim=2)
        r22 = w2 + torch.cross(w1, w3, dim=2)
        r23 = w3 + torch.cross(w2, w1, dim=2)
        recoverR2 = torch.cat([r21, r22, r23], dim=3)

        recoverR1 = recoverR1 * torch.sign(torch.det(recoverR1)).unsqueeze(-1).unsqueeze(-1).expand([-1,-1,3,3])
        recoverR2 = recoverR2 * torch.sign(torch.det(recoverR2)).unsqueeze(-1).unsqueeze(-1).expand([-1,-1,3,3])

        combs_t = [t, t, -t, -t]
        combs_R = [recoverR1, recoverR2, recoverR1, recoverR2]

        return [combs_t, combs_R]

    def select_RT(self, combinations, pts2d1, pts2d2, intrinsic, insmap):
        combs_t, combs_R = combinations
        pospts = list()

        with torch.no_grad():
            for k in range(4):
                tc = combs_t[k]
                Rc = combs_R[k]
                rdepth1 = self.flow2depth_relative(pts2d1, pts2d2, intrinsic, Rc, tc, insmap)
                rdepth2 = self.flow2depth_relative(pts2d2, pts2d1, intrinsic, torch.transpose(Rc, dim0=2, dim1=3), -tc, insmap)
                posnum = (rdepth1 > 0).float() + (rdepth2 > 0).float()
                posnum_ins = self.eppcompress(insmap, posnum, self.maxinsnum)
                pospts.append(posnum_ins)
            pospts = torch.cat(pospts, dim=2)
            maxidx = torch.argmax(pospts, dim=2, keepdim=True).squeeze(-1).squeeze(-1).int()

        choiced_t = self.eppsel(maxidx, torch.stack(combs_t, dim=2))
        choiced_R = self.eppsel(maxidx, torch.stack(combs_R, dim=2))
        return choiced_t, choiced_R

    def select_RT_uguess(self, combinations, poses_guess):
        combs_t, combs_R = combinations
        combs_t_stack = torch.stack(combs_t, dim=2)
        combs_R_stack = torch.stack(combs_R, dim=2)
        with torch.no_grad():
            t_guess = poses_guess[:, :, 0:3, 3:4] / torch.norm(poses_guess[:, :, 0:3, 3:4], dim=[2, 3], keepdim=True)
            R = poses_guess[:, :, 0:3, 0:3]

            t_diff = (combs_t_stack - t_guess.unsqueeze(2).expand([-1, -1, 4, -1, -1])).abs().sum(dim=[3,4])
            R_diff = (combs_R_stack - R.unsqueeze(2).expand([-1, -1, 4, -1, -1])).abs().sum(dim=[3,4])
            minidx = torch.argmin(t_diff + R_diff, dim=2, keepdim=True).squeeze(-1).squeeze(-1).int()

        choiced_t = self.eppsel(minidx, combs_t_stack)
        choiced_R = self.eppsel(minidx, combs_R_stack)

        return choiced_t, choiced_R

    def flow2depth_relative(self, pts2d1, pts2d2, intrinsic, R, t, insmap):
        M = intrinsic @ R @ torch.inverse(intrinsic)
        delta_t = intrinsic @ t

        M_inf = self.eppinflate(insmap, M)
        delta_t_inf = self.eppinflate(insmap, delta_t)

        Mx_inf, My_inf, Mz_inf = torch.split(M_inf, 1, dim=3)
        delta_tx_inf, delta_ty_inf, delta_tz_inf = torch.split(delta_t_inf, 1, dim=3)
        pts2d2x, pts2d2y, _ = torch.split(pts2d2, 1, dim=3)

        denom = (pts2d2x * (Mz_inf @ pts2d1) - (Mx_inf @ pts2d1)) + (pts2d2y * (Mz_inf @ pts2d1) - (My_inf @ pts2d1))
        rdepth = ((delta_tx_inf - pts2d2x * delta_tz_inf) + (delta_ty_inf - pts2d2y * delta_tz_inf)) / denom
        return rdepth

    def flow2epp(self, insmap, flowmap, intrinsic, t_init=None, ang_init=None, poses_guess=None):
        if t_init is None:
            t_init = self.t_init
        if ang_init is None:
            ang_init = self.ang_init

        inscount = self.eppcompress(insmap, (insmap > -1).squeeze(1).unsqueeze(-1).unsqueeze(-1).float(), self.maxinsnum)
        inscount_inf = self.eppinflate(insmap, inscount) + 1e-5

        intrinsic_ex = intrinsic.unsqueeze(1).expand([-1, self.maxinsnum, -1, -1])
        intrinsic_inv = torch.inverse(intrinsic)
        intrinsic_inv_inf = intrinsic_inv.view([self.bz, 1, 1, 3, 3]).expand([-1, self.height, self.width, -1, -1])

        pts1, pts2 = self.flowmap2pts(flowmap)
        ptsl = torch.transpose(torch.transpose(pts2, dim0=3, dim1=4) @ torch.transpose(intrinsic_inv_inf, dim0=3, dim1=4), dim0=3, dim1=4)
        ptsr = intrinsic_inv_inf @ pts1
        derivM = ptsl @ torch.transpose(ptsr, dim0=3, dim1=4)

        r_last = torch.ones_like(inscount) * 1e10
        terminateflag = torch.ones_like(inscount, dtype=torch.bool) * (inscount > 0)

        t = t_init
        ang = ang_init
        for k in range(self.itnum):
            T = self.t2T(t)
            R, rotxd, rotyd, rotzd = self.ang2R(ang, requires_deriv=True)

            T_inf = self.eppinflate(insmap, T)
            R_inf = self.eppinflate(insmap, R)
            rotxd_inf = self.eppinflate(insmap, rotxd)
            rotyd_inf = self.eppinflate(insmap, rotyd)
            rotzd_inf = self.eppinflate(insmap, rotzd)

            tnorm = torch.norm(t, dim=[2, 3], keepdim=True)
            J_t0_bias = 2 * self.lap * (tnorm - 1) / tnorm * t[:, :, 0:1, :]
            J_t1_bias = 2 * self.lap * (tnorm - 1) / tnorm * t[:, :, 1:2, :]
            J_t2_bias = 2 * self.lap * (tnorm - 1) / tnorm * t[:, :, 2:3, :]
            J_t0_bias_inf = self.eppinflate(insmap, J_t0_bias)
            J_t1_bias_inf = self.eppinflate(insmap, J_t1_bias)
            J_t2_bias_inf = self.eppinflate(insmap, J_t2_bias)
            tnorm_inf = self.eppinflate(insmap, tnorm)

            r = (torch.transpose(ptsl, dim0=3, dim1=4) @ T_inf @ R_inf @ ptsr)
            r_square = (r ** 2 + self.lap * (tnorm_inf - 1) ** 2) / inscount_inf

            J_t0 = (torch.sum(derivM * (self.T0_inf @ R_inf), dim=[3, 4], keepdim=True) * 2 * r + J_t0_bias_inf) / inscount_inf
            J_t1 = (torch.sum(derivM * (self.T1_inf @ R_inf), dim=[3, 4], keepdim=True) * 2 * r + J_t1_bias_inf) / inscount_inf
            J_t2 = (torch.sum(derivM * (self.T2_inf @ R_inf), dim=[3, 4], keepdim=True) * 2 * r + J_t2_bias_inf) / inscount_inf

            J_ang0 = (torch.sum(derivM * (T_inf @ rotxd_inf), dim=[3, 4], keepdim=True) * 2 * r) / inscount_inf
            J_ang1 = (torch.sum(derivM * (T_inf @ rotyd_inf), dim=[3, 4], keepdim=True) * 2 * r) / inscount_inf
            J_ang2 = (torch.sum(derivM * (T_inf @ rotzd_inf), dim=[3, 4], keepdim=True) * 2 * r) / inscount_inf

            JacobM = torch.cat([J_ang0, J_ang1, J_ang2, J_t0, J_t1, J_t2], dim=-2)
            MM = JacobM @ torch.transpose(JacobM, dim1=3, dim0=4)
            updateMl = self.eppcompress(insmap, MM, self.maxinsnum)
            Mr = JacobM * r_square.expand([-1,-1,-1,6,-1])
            updateMr = self.eppcompress(insmap, Mr, self.maxinsnum)

            r_current = self.eppcompress(insmap, r_square, self.maxinsnum)

            updateang, updatel, terminateflag = self.eppupdate(updateMl, updateMr, inscount, r_current, r_last, terminateflag, self.idtpadding)
            r_last = r_current
            if updateang is None or updatel is None:
                break

            t = t - updatel * self.lr
            ang = ang - updateang * self.lr

        t = t / torch.norm(t, dim=[2,3], keepdim=True)

        if poses_guess is None:
            t, R = self.select_RT(combinations=self.extract_tR_combs(t, ang), pts2d1=pts1, pts2d2=pts2, intrinsic=intrinsic_ex, insmap=insmap)
        else:
            t, R = self.select_RT_uguess(combinations=self.extract_tR_combs(t, ang), poses_guess=poses_guess)

        return t, R

    def flow2epp_analysis(self, insmap, flowmap, intrinsic, bz, insidx, t_init=None, ang_init=None, poses_guess=None):
        if t_init is None:
            t_init = self.t_init
        if ang_init is None:
            ang_init = self.ang_init

        inscount = self.eppcompress(insmap, (insmap > -1).squeeze(1).unsqueeze(-1).unsqueeze(-1).float(), self.maxinsnum)
        inscount_inf = self.eppinflate(insmap, inscount) + 1e-5

        intrinsic_ex = intrinsic.unsqueeze(1).expand([-1, self.maxinsnum, -1, -1])
        intrinsic_inv = torch.inverse(intrinsic)
        intrinsic_inv_inf = intrinsic_inv.view([self.bz, 1, 1, 3, 3]).expand([-1, self.height, self.width, -1, -1])

        pts1, pts2 = self.flowmap2pts(flowmap)
        ptsl = torch.transpose(torch.transpose(pts2, dim0=3, dim1=4) @ torch.transpose(intrinsic_inv_inf, dim0=3, dim1=4), dim0=3, dim1=4)
        ptsr = intrinsic_inv_inf @ pts1
        derivM = ptsl @ torch.transpose(ptsr, dim0=3, dim1=4)

        r_last = torch.ones_like(inscount) * 1e10
        terminateflag = torch.ones_like(inscount, dtype=torch.bool) * (inscount > 0)

        t = t_init
        ang = ang_init
        for k in range(self.itnum):
            T = self.t2T(t)
            R, rotxd, rotyd, rotzd = self.ang2R(ang, requires_deriv=True)

            T_inf = self.eppinflate(insmap, T)
            R_inf = self.eppinflate(insmap, R)
            rotxd_inf = self.eppinflate(insmap, rotxd)
            rotyd_inf = self.eppinflate(insmap, rotyd)
            rotzd_inf = self.eppinflate(insmap, rotzd)

            tnorm = torch.norm(t, dim=[2, 3], keepdim=True)
            J_t0_bias = 2 * self.lap * (tnorm - 1) / tnorm * t[:, :, 0:1, :]
            J_t1_bias = 2 * self.lap * (tnorm - 1) / tnorm * t[:, :, 1:2, :]
            J_t2_bias = 2 * self.lap * (tnorm - 1) / tnorm * t[:, :, 2:3, :]
            J_t0_bias_inf = self.eppinflate(insmap, J_t0_bias)
            J_t1_bias_inf = self.eppinflate(insmap, J_t1_bias)
            J_t2_bias_inf = self.eppinflate(insmap, J_t2_bias)
            tnorm_inf = self.eppinflate(insmap, tnorm)

            r = (torch.transpose(ptsl, dim0=3, dim1=4) @ T_inf @ R_inf @ ptsr)
            r_square = (r ** 2 + self.lap * (tnorm_inf - 1) ** 2) / inscount_inf

            J_t0 = (torch.sum(derivM * (self.T0_inf @ R_inf), dim=[3, 4], keepdim=True) * 2 * r + J_t0_bias_inf) / inscount_inf
            J_t1 = (torch.sum(derivM * (self.T1_inf @ R_inf), dim=[3, 4], keepdim=True) * 2 * r + J_t1_bias_inf) / inscount_inf
            J_t2 = (torch.sum(derivM * (self.T2_inf @ R_inf), dim=[3, 4], keepdim=True) * 2 * r + J_t2_bias_inf) / inscount_inf

            J_ang0 = (torch.sum(derivM * (T_inf @ rotxd_inf), dim=[3, 4], keepdim=True) * 2 * r) / inscount_inf
            J_ang1 = (torch.sum(derivM * (T_inf @ rotyd_inf), dim=[3, 4], keepdim=True) * 2 * r) / inscount_inf
            J_ang2 = (torch.sum(derivM * (T_inf @ rotzd_inf), dim=[3, 4], keepdim=True) * 2 * r) / inscount_inf

            JacobM = torch.cat([J_ang0, J_ang1, J_ang2, J_t0, J_t1, J_t2], dim=-2)
            MM = JacobM @ torch.transpose(JacobM, dim1=3, dim0=4)
            updateMl = self.eppcompress(insmap, MM, self.maxinsnum)
            Mr = JacobM * r_square.expand([-1,-1,-1,6,-1])
            updateMr = self.eppcompress(insmap, Mr, self.maxinsnum)

            r_current = self.eppcompress(insmap, r ** 2, self.maxinsnum)

            updateang, updatel, terminateflag = self.eppupdate(updateMl, updateMr, inscount, r_current, r_last, terminateflag, self.idtpadding)

            if terminateflag is not None:
                if terminateflag[bz, insidx, 0, 0] == 1:
                    print("Iteration %d, loss: %.2E" % (k, r_current[bz, insidx, 0, 0].detach().cpu().item()))

            r_last = r_current
            if updateang is None or updatel is None:
                break

            t = t - updatel * self.lr
            ang = ang - updateang * self.lr

        t = t / torch.norm(t, dim=[2,3], keepdim=True)

        if poses_guess is None:
            t, R = self.select_RT(combinations=self.extract_tR_combs(t, ang), pts2d1=pts1, pts2d2=pts2, intrinsic=intrinsic_ex, insmap=insmap)
        else:
            t, R = self.select_RT_uguess(combinations=self.extract_tR_combs(t, ang), poses_guess=poses_guess)

        return t, R

    def ck_tRcombs_correctness(self, combs):
        t_combs, R_combs = combs
        for k in range(4):
            T = self.t2T(t_combs[k])
            R = R_combs[k]
            print((T @ R).abs().sum(dim=[2, 3])[0, 0:3])

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

def debug_backward_inflate():
    w = 1241
    h = 376
    bz = 2

    infh = 6
    infw = 6
    maxinsnum = 100
    instance = torch.randint(0, 100, [bz, 1, h, w], dtype=torch.int32, device=torch.device(type='cuda', index=0))
    infsrc = (torch.rand([bz, maxinsnum, infh, infw], dtype=torch.float32, device=torch.device(type='cuda', index=0)) - 0.5) * 2
    infsrc.requires_grad = True

    eppinflate = eppcore_inflation.apply
    infdst = eppinflate(instance, infsrc)

    def lossfunc(input):
        return (input.abs()).sum()

    loss = lossfunc(infdst)
    loss.backward()

    with torch.no_grad():
        ckw = torch.randint(0, infw, [1]).item()
        ckh = torch.randint(0, infh, [1]).item()
        ckbz = torch.randint(0, bz, [1]).item()
        ckins = torch.randint(0, maxinsnum, [1]).item()

        delta = 1e-2

        infsrc_plus = torch.clone(infsrc).detach()
        infsrc_neg = torch.clone(infsrc).detach()
        infsrc_plus[ckbz, ckins, ckh, ckw] = infsrc_plus[ckbz, ckins, ckh, ckw] + delta
        infsrc_neg[ckbz, ckins, ckh, ckw] = infsrc_neg[ckbz, ckins, ckh, ckw] - delta

        infdst_plus = eppinflate(instance, infsrc_plus)
        infdst_neg = eppinflate(instance, infsrc_neg)

        numgrad = (lossfunc(infdst_plus) - lossfunc(infdst_neg)) / delta / 2
        theograd = infsrc.grad[ckbz, ckins, ckh, ckw]

        print("Numerical: %f, theoretical: %f" % (numgrad.item(), theograd.item()))

def debug_backward_compress():
    w = 1241
    h = 376
    bz = 2

    infh = 6
    infw = 6
    maxinsnum = 100
    instance = torch.randint(0, 100, [bz, 1, h, w], dtype=torch.int32, device=torch.device(type='cuda', index=0))
    compresssrc = torch.rand([bz, h, w, infh, infw], dtype=torch.float32, device=torch.device(type='cuda', index=0), requires_grad=True)

    eppcompress = eppcore_compression.apply
    compressdst = eppcompress(instance, compresssrc, maxinsnum)

    def lossfunc(input):
        return (input.abs()).sum()

    loss = lossfunc(compressdst)
    loss.backward()

    with torch.no_grad():
        ckw = torch.randint(0, infw, [1]).item()
        ckh = torch.randint(0, infh, [1]).item()
        ckbz = torch.randint(0, bz, [1]).item()
        ckww = torch.randint(0, w, [1]).item()
        ckhh = torch.randint(0, h, [1]).item()

        delta = 1

        compsrc_plus = torch.clone(compresssrc).detach()
        compsrc_neg = torch.clone(compresssrc).detach()
        compsrc_plus[ckbz, ckhh, ckww, ckh, ckw] = compsrc_plus[ckbz, ckhh, ckww, ckh, ckw] + delta
        compsrc_neg[ckbz, ckhh, ckww, ckh, ckw] = compsrc_neg[ckbz, ckhh, ckww, ckh, ckw] - delta

        compdst_plus = eppcompress(instance, compsrc_plus, maxinsnum)
        compdst_neg = eppcompress(instance, compsrc_neg, maxinsnum)

        numgrad = (lossfunc(compdst_plus) - lossfunc(compdst_neg)) / 2 / delta
        theograd = compresssrc.grad[ckbz, ckhh, ckww, ckh, ckw]

        print("Numerical: %f, theoretical: %f" % (numgrad.item(), theograd.item()))

if __name__ == '__main__':
    debug_epp_batchselection()
    debug_backward_inflate()
    debug_backward_compress()

