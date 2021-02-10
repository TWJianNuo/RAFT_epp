import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))) # location of src

import numpy as np
from PIL import Image, ImageFile
import torch
import glob
import math
import copy
import pickle
from kittiexp.semantic_labels import *
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_intrinsic(calibpath):
    cam2cam = read_calib_file(calibpath)
    K = np.eye(4, dtype=np.float32)
    K[0:3, :] = cam2cam['P_rect_02'].reshape(3, 4)
    return K

def readlines(filename):
    with open(filename, 'r') as f:
        filenames = f.readlines()
    return filenames

def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data

def generate_seqmapping():
    seqmapping = \
    ['00 2011_10_03_drive_0027 000000 004540',
     '01 2011_10_03_drive_0042 000000 001100',
     "02 2011_10_03_drive_0034 000000 004660",
     "03 2011_09_26_drive_0067 000000 000800",
     "04 2011_09_30_drive_0016 000000 000270",
     "05 2011_09_30_drive_0018 000000 002760",
     "06 2011_09_30_drive_0020 000000 001100",
     "07 2011_09_30_drive_0027 000000 001100",
     "08 2011_09_30_drive_0028 001100 005170",
     "09 2011_09_30_drive_0033 000000 001590",
     "10 2011_09_30_drive_0034 000000 001200"]

    seqmap = dict()
    for seqm in seqmapping:
        mapentry = dict()
        mapid, seqname, stid, enid = seqm.split(' ')
        mapentry['mapid'] = int(mapid)
        mapentry['stid'] = int(stid)
        mapentry['enid'] = int(enid)
        seqmap[seqname] = mapentry
    return seqmap

def get_all_img_entry(predroot):
    predpaths = list()
    days = glob.glob(os.path.join(predroot, '*/'))
    for d in days:
        seqs = glob.glob(os.path.join(d, '*/'))
        for seq in seqs:
            pngpaths = glob.glob(os.path.join(seq, '*.npy'))
            predpaths = predpaths + pngpaths
    predpaths.sort()

    import random
    random.seed(0)
    random.shuffle(predpaths)
    return predpaths

def acquire_gt_flow(intrinsics, depthgt, posesgt, semantic_selector_arr):
    intrinsics = intrinsics.numpy()
    depthgt = depthgt.numpy()
    selecotr = (depthgt > 0) * (semantic_selector_arr == 1)
    h, w = depthgt.shape
    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
    xxf = xx[selecotr]
    yyf = yy[selecotr]

    depthf = depthgt[selecotr]
    pts3d = np.stack([xxf * depthf, yyf * depthf, depthf, np.ones_like(depthf)], axis=0)
    pts3d = np.linalg.inv(intrinsics) @ pts3d
    pts3doview = posesgt @ pts3d
    pts2doview = intrinsics @ copy.deepcopy(pts3doview)
    pts2doview[0, :] = pts2doview[0, :] / pts2doview[2, :]
    pts2doview[1, :] = pts2doview[1, :] / pts2doview[2, :]
    pts2doview = pts2doview[0:2, :]

    pts2d1 = torch.from_numpy(np.stack([xxf, yyf, np.ones_like(xxf)], axis=0)).float()
    pts2d2 = torch.from_numpy(np.stack([pts2doview[0, :], pts2doview[1, :], np.ones_like(xxf)], axis=0)).float()
    return pts2d1, pts2d2

class eppConcluer(torch.nn.Module):
    def __init__(self, itnum=2, laplacian=1e-2, lr=1):
        super(eppConcluer, self).__init__()
        self.itnum = itnum
        self.laplacian = laplacian
        self.lr = lr

    def rot2ang(self, R):
        sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        ang = torch.zeros([3], device=R.device, dtype=torch.float)
        ang[0] = torch.atan2(R[2, 1], R[2, 2])
        ang[1] = torch.atan2(-R[2, 0], sy)
        ang[2] = torch.atan2(R[1, 0], R[0, 0])
        return ang

    def rot_from_axisangle(self, angs):
        """Convert an axisangle rotation into a 4x4 transformation matrix
        (adapted from https://github.com/Wallacoloo/printipi)
        Input 'vec' has to be Bx1x3
        """
        rotx = torch.eye(3, device=angs.device, dtype=torch.float)
        roty = torch.eye(3, device=angs.device, dtype=torch.float)
        rotz = torch.eye(3, device=angs.device, dtype=torch.float)

        rotx[1, 1] = torch.cos(angs[0])
        rotx[1, 2] = -torch.sin(angs[0])
        rotx[2, 1] = torch.sin(angs[0])
        rotx[2, 2] = torch.cos(angs[0])

        roty[0, 0] = torch.cos(angs[1])
        roty[0, 2] = torch.sin(angs[1])
        roty[2, 0] = -torch.sin(angs[1])
        roty[2, 2] = torch.cos(angs[1])

        rotz[0, 0] = torch.cos(angs[2])
        rotz[0, 1] = -torch.sin(angs[2])
        rotz[1, 0] = torch.sin(angs[2])
        rotz[1, 1] = torch.cos(angs[2])

        rot = rotz @ (roty @ rotx)
        return rot

    def t2T(self, t):
        T = torch.zeros([3, 3], device=t.device, dtype=torch.float)
        T[0, 1] = -t[2]
        T[0, 2] = t[1]
        T[1, 0] = t[2]
        T[1, 2] = -t[0]
        T[2, 0] = -t[1]
        T[2, 1] = t[0]
        return T

    def derivative_angle(self, angs):
        rotx = torch.eye(3, device=angs.device, dtype=torch.float)
        roty = torch.eye(3, device=angs.device, dtype=torch.float)
        rotz = torch.eye(3, device=angs.device, dtype=torch.float)

        rotx[1, 1] = torch.cos(angs[0])
        rotx[1, 2] = -torch.sin(angs[0])
        rotx[2, 1] = torch.sin(angs[0])
        rotx[2, 2] = torch.cos(angs[0])

        roty[0, 0] = torch.cos(angs[1])
        roty[0, 2] = torch.sin(angs[1])
        roty[2, 0] = -torch.sin(angs[1])
        roty[2, 2] = torch.cos(angs[1])

        rotz[0, 0] = torch.cos(angs[2])
        rotz[0, 1] = -torch.sin(angs[2])
        rotz[1, 0] = torch.sin(angs[2])
        rotz[1, 1] = torch.cos(angs[2])

        rotxd = torch.zeros([3, 3], device=angs.device, dtype=torch.float)
        rotyd = torch.zeros([3, 3], device=angs.device, dtype=torch.float)
        rotzd = torch.zeros([3, 3], device=angs.device, dtype=torch.float)

        rotxd[1, 1] = -torch.sin(angs[0])
        rotxd[1, 2] = -torch.cos(angs[0])
        rotxd[2, 1] = torch.cos(angs[0])
        rotxd[2, 2] = -torch.sin(angs[0])

        rotyd[0, 0] = -torch.sin(angs[1])
        rotyd[0, 2] = torch.cos(angs[1])
        rotyd[2, 0] = -torch.cos(angs[1])
        rotyd[2, 2] = -torch.sin(angs[1])

        rotzd[0, 0] = -torch.sin(angs[2])
        rotzd[0, 1] = -torch.cos(angs[2])
        rotzd[1, 0] = torch.cos(angs[2])
        rotzd[1, 1] = -torch.sin(angs[2])

        rotxd = rotz @ roty @ rotxd
        rotyd = rotz @ rotyd @ rotx
        rotzd = rotzd @ roty @ rotx

        return rotxd, rotyd, rotzd

    def derivative_translate(self, device):
        T0 = torch.zeros([3, 3], device=device, dtype=torch.float)
        T1 = torch.zeros([3, 3], device=device, dtype=torch.float)
        T2 = torch.zeros([3, 3], device=device, dtype=torch.float)

        T0[1, 2] = -1
        T0[2, 1] = 1

        T1[0, 2] = 1
        T1[2, 0] = -1

        T2[0, 1] = -1
        T2[1, 0] = 1

        return T0, T1, T2

    def compute_JacobianM(self, pts2d1, pts2d2, intrinsic, t, ang, lagr):
        R = self.rot_from_axisangle(ang)
        T = self.t2T(t)

        derT0, derT1, derT2 = self.derivative_translate(intrinsic.device)
        rotxd, rotyd, rotzd = self.derivative_angle(ang)

        pts2d1_bz = (pts2d1.T).unsqueeze(2)
        pts2d2_bz = (pts2d2.T).unsqueeze(2)
        samplenum = pts2d1.shape[1]

        r_bias = (torch.norm(t) - 1)
        J_t0_bias = 2 * lagr * r_bias / torch.norm(t) * t[0]
        J_t1_bias = 2 * lagr * r_bias / torch.norm(t) * t[1]
        J_t2_bias = 2 * lagr * r_bias / torch.norm(t) * t[2]

        pts2d2_bz_t = torch.transpose(torch.transpose(pts2d2_bz, 1, 2) @ torch.inverse(intrinsic).T, 1, 2)
        pts2d1_bz_t = torch.inverse(intrinsic) @ pts2d1_bz
        r = (torch.transpose(pts2d2_bz_t, 1, 2) @ T @ R @ pts2d1_bz_t).squeeze()
        derivM = pts2d2_bz_t @ torch.transpose(pts2d1_bz_t, 1, 2)

        J_t0 = torch.sum(derivM * (derT0 @ R), dim=[1, 2]) * 2 * r / samplenum + J_t0_bias / samplenum
        J_t1 = torch.sum(derivM * (derT1 @ R), dim=[1, 2]) * 2 * r / samplenum + J_t1_bias / samplenum
        J_t2 = torch.sum(derivM * (derT2 @ R), dim=[1, 2]) * 2 * r / samplenum + J_t2_bias / samplenum

        J_ang0 = torch.sum(derivM * (T @ rotxd), dim=[1, 2]) * 2 * r / samplenum
        J_ang1 = torch.sum(derivM * (T @ rotyd), dim=[1, 2]) * 2 * r / samplenum
        J_ang2 = torch.sum(derivM * (T @ rotzd), dim=[1, 2]) * 2 * r / samplenum

        JacobM = torch.stack([J_ang0, J_ang1, J_ang2, J_t0, J_t1, J_t2], dim=1)
        residual = (r ** 2 / samplenum + lagr * r_bias ** 2 / samplenum).unsqueeze(1)
        return JacobM, residual, r.abs().mean().detach()

    def newton_gauss_F(self, pts2d1, pts2d2, intrinsic, posegt, depthgt):
        # Newton Gauss Alg
        outputsrec = dict()

        intrinsic = intrinsic[0:3, 0:3]

        ang_reg = torch.zeros([3], device=intrinsic.device, dtype=torch.float)
        t_reg = torch.zeros([3], device=intrinsic.device, dtype=torch.float)
        t_reg[-1] = -0.99

        E_init = self.t2T(t_reg / torch.norm(t_reg)) @ self.rot_from_axisangle(ang_reg)
        F_init = torch.inverse(intrinsic).T @ E_init @ torch.inverse(intrinsic)
        loss_init = float(torch.mean(torch.abs(torch.sum((pts2d2.T @ F_init) * pts2d1.T, dim=1))).cpu().numpy())

        minLoss = 1e10

        import time
        sttime = time.time()
        sumt1 = 0

        for kk in range(self.itnum):
            tmpst = time.time()
            J, r, totloss = self.compute_JacobianM(pts2d1, pts2d2, intrinsic, t_reg, ang_reg, self.laplacian)
            sumt1 += time.time() - tmpst

            curloss = r.sum().detach()
            try:
                M = J.T @ J
                inverseM = torch.inverse(M)

                if (inverseM @ M - torch.eye(6, device=intrinsic.device, dtype=torch.float)).abs().max() > 0.1:
                    break
                updatem = inverseM @ J.T @ r
            except:
                break

            if curloss > minLoss:
                break
            else:
                minLoss = curloss

            ang_reg = (ang_reg - self.lr * updatem[0:3, 0])
            t_reg = (t_reg - self.lr * updatem[3:6, 0])

        timeprop = sumt1 / (time.time() - sttime)

        E_est = self.t2T(t_reg / torch.norm(t_reg)) @ self.rot_from_axisangle(ang_reg)
        F_est = torch.inverse(intrinsic).T @ E_est @ torch.inverse(intrinsic)
        loss_est = float(torch.mean(torch.abs(torch.sum((pts2d2.T @ F_est) * pts2d1.T, dim=1))).cpu().numpy())

        E_gt = self.t2T((posegt[0:3, 3]) / torch.norm(posegt[0:3, 3])) @ posegt[0:3, 0:3]
        F_gt = torch.inverse(intrinsic).T @ E_gt @ torch.inverse(intrinsic)
        loss_gt = torch.mean(torch.abs(torch.sum((pts2d2.T @ F_gt) * pts2d1.T, dim=1)))

        t_gt = (posegt[0:3, 3] / torch.norm(posegt[0:3, 3])).float()

        t_est, Rpred = self.select_RT(self.extract_RT_analytic(E_est), pts2d1, pts2d2, intrinsic)
        ang_reg = self.rot2ang(Rpred)

        loss_mv = 1 - torch.sum(t_est * t_gt)
        loss_ang = torch.mean((self.rot2ang((posegt[0:3, 0:3])) - ang_reg).abs())

        # rdepth, alpha, selector = self.flow2depth(pts2d1, pts2d2, intrinsic, Rpred, t_est, depthgt)

        # depthdiff = torch.abs(rdepth[selector] - depthgt[selector]).mean()
        outputsrec['loss_mv'] = float(loss_mv.detach().cpu().numpy())
        outputsrec['loss_ang'] = float(loss_ang.detach().cpu().numpy())
        # outputsrec['loss_depth'] = float(depthdiff.detach().cpu().numpy())
        outputsrec['loss_constrain'] = loss_est
        # print(
        #     "Optimization finished at step %d, mv loss: %f, ang loss: %f, depthloss: %f, norm: %f, in all points: %f, gt pose: %f" % (
        #     kk, loss_mv, loss_ang, depthdiff, torch.norm(t_reg), loss_est, loss_gt))
        print("Optimization finished at step %d, mv loss: %f, ang loss: %f, norm: %f, in all points: %f, gt pose: %f, init: %f, time prop: %f" % (kk, loss_mv, loss_ang, torch.norm(t_reg), loss_est, loss_gt, loss_init, timeprop))
        return outputsrec

    def extract_RT_analytic(self, E):
        w = torch.zeros([3, 3], device=E.device, dtype=torch.float)
        w[0, 1] = -1
        w[1, 0] = 1
        w[2, 2] = 1

        M = E @ E.T
        t2 = torch.sqrt((M[0, 0] + M[1, 1] - M[2, 2]) / 2)
        t1 = torch.exp(torch.log(torch.abs((M[1, 2] + M[2, 1]) / 2)) - torch.log(torch.abs(t2))) * torch.sign(t2) * torch.sign(M[2, 1] + M[1, 2]) * (-1)
        t0 = torch.exp(torch.log(torch.abs((M[0, 2] + M[2, 0]) / 2)) - torch.log(torch.abs(t2))) * torch.sign(t2) * torch.sign(M[2, 0] + M[0, 2]) * (-1)
        recovert = torch.stack([t0, t1, t2])

        w1 = torch.cross(recovert, E[:, 0])
        w2 = torch.cross(recovert, E[:, 1])
        w3 = torch.cross(recovert, E[:, 2])

        r11 = w1 + torch.cross(w2, w3)
        r12 = w2 + torch.cross(w3, w1)
        r13 = w3 + torch.cross(w1, w2)
        recoverR1 = torch.stack([r11, r12, r13], dim=0).T

        r21 = w1 + torch.cross(w3, w2)
        r22 = w2 + torch.cross(w1, w3)
        r23 = w3 + torch.cross(w2, w1)
        recoverR2 = torch.stack([r21, r22, r23], dim=0).T

        if torch.det(recoverR1) < 0:
            recoverR1 = -recoverR1
        if torch.det(recoverR2) < 0:
            recoverR2 = -recoverR2

        combinations = [
            [recovert, recoverR1],
            [recovert, recoverR2],
            [-recovert, recoverR1],
            [-recovert, recoverR2]
        ]

        return combinations

    def select_RT(self, combinations, pts2d1, pts2d2, intrinsic):
        pospts = torch.zeros([4], device=intrinsic.device, dtype=torch.float)
        for idx, (tc, Rc) in enumerate(combinations):
            rdepth1 = self.flow2depth_relative(pts2d1, pts2d2, intrinsic, Rc, tc)
            rdepth2 = self.flow2depth_relative(pts2d2, pts2d1, intrinsic, Rc.T, -tc)
            pospts[idx] = torch.sum(rdepth1 > 0) + torch.sum(rdepth2 > 0)
        maxidx = torch.argmax(pospts)
        return combinations[maxidx]

    def flow2depth_relative(self, pts2d1, pts2d2, intrinsic, R, t):
        M = intrinsic @ R @ torch.inverse(intrinsic)
        delta_t = (intrinsic @ t.unsqueeze(1)).squeeze()

        denom = (pts2d2[0, :] * (M[2, :].unsqueeze(0) @ pts2d1).squeeze() - (M[0, :].unsqueeze(0) @ pts2d1).squeeze()) + \
                (pts2d2[1, :] * (M[2, :].unsqueeze(0) @ pts2d1).squeeze() - (M[1, :].unsqueeze(0) @ pts2d1).squeeze())
        rdepth = ((delta_t[0] - pts2d2[0, :] * delta_t[2]) + (delta_t[1] - pts2d2[1, :] * delta_t[2])) / denom
        return rdepth

    def flow2depth(self, pts2d1, pts2d2, intrinsic, R, t, coorespondedDepth):
        M = intrinsic @ R @ torch.inverse(intrinsic)
        delta_t = (intrinsic @ t.unsqueeze(1)).squeeze()
        minval = 1e-6

        denom = (pts2d2[0, :] * (M[2, :].unsqueeze(0) @ pts2d1).squeeze() - (M[0, :].unsqueeze(0) @ pts2d1).squeeze()) ** 2 + \
                (pts2d2[1, :] * (M[2, :].unsqueeze(0) @ pts2d1).squeeze() - (M[1, :].unsqueeze(0) @ pts2d1).squeeze()) ** 2

        selector = (denom > minval)
        denom = torch.clamp(denom, min=minval, max=np.inf)

        rel_d = torch.sqrt(
            ((delta_t[0] - pts2d2[0, :] * delta_t[2]) ** 2 +
             (delta_t[1] - pts2d2[1, :] * delta_t[2]) ** 2) / denom)
        alpha = torch.mean(coorespondedDepth) / torch.mean(rel_d)
        recover_d = alpha * rel_d

        return recover_d, alpha, selector


rawroot = '/media/shengjie/disk1/data/Kitti'
depthroot = '/home/shengjie/Documents/Data/Kitti/filtered_lidar'
flowpre_root = '/media/shengjie/disk1/Prediction/RAFT_odom_flow'
odomgtroot = '/media/shengjie/disk1/Prediction/odometryseq/dataset/poses'
exportroot = '/media/shengjie/disk1/Prediction/Deepv2d_odom'
svroot = '/media/shengjie/disk1/Prediction/GaussNewtonM_pred'
semanticroot = '/home/shengjie/Documents/Data/Kitti/kitti_predSemantics'
seqmap = generate_seqmapping()
flowpre_paths = get_all_img_entry(flowpre_root)

eppconstrain = list()
angloss = list()
mvloss = list()
depthloss = list()

eppcocluer = eppConcluer()
with torch.no_grad():
    for flowpre_path in flowpre_paths:
        flowpred = np.load(flowpre_path)

        day, seq, imgname = flowpre_path.split('/')[-3::]
        imgname = imgname.replace("npy", 'png')
        frameidx = int(imgname.split('.')[0])

        svfold = os.path.join(svroot, day, seq)
        os.makedirs(svfold, exist_ok=True)
        svpath = os.path.join(svfold, str(frameidx).zfill(10) + '.pickle')

        validmark = os.path.exists(os.path.join(rawroot, day, seq, 'image_02', 'data', "{}.png".format(str(int(frameidx)).zfill(10)))) and \
                    os.path.exists(os.path.join(rawroot, day, seq, 'image_02', 'data', "{}.png".format(str(int(frameidx + 1)).zfill(10)))) and \
                    os.path.exists(os.path.join(depthroot, day, seq, 'image_02', "{}.png".format(str(int(frameidx)).zfill(10)))) and \
                    os.path.exists(os.path.join(odomgtroot, "{}.txt".format(str(seqmap[seq[0:21]]['mapid']).zfill(2)))) and \
                    os.path.exists(os.path.join(exportroot, day, seq, "posepred/{}.txt".format(str(frameidx).zfill(10))))

        if validmark == False:
            continue

        rgb = Image.open(os.path.join(rawroot, day, seq, 'image_02', 'data', "{}.png".format(str(int(frameidx)).zfill(10))))
        rgbn = Image.open(os.path.join(rawroot, day, seq, 'image_02', 'data', "{}.png".format(str(int(frameidx + 1)).zfill(10))))
        depthmap = np.array(Image.open(os.path.join(depthroot, day, seq, 'image_02', "{}.png".format(str(int(frameidx)).zfill(10))))).astype(np.float32) / 256.0
        depthmapf = torch.from_numpy(depthmap.flatten()).float()
        intrinsic = get_intrinsic(os.path.join(rawroot, day, 'calib_cam_to_cam.txt'))
        intrinsic[0:3, 3] = 0

        depthmap = torch.from_numpy(depthmap)
        intrinsic = torch.from_numpy(intrinsic)

        w, h = rgb.size

        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        pts2d1 = torch.stack([torch.from_numpy(xx.flatten()), torch.from_numpy(yy.flatten()), torch.ones(int(w * h))], dim=0).contiguous().float()
        pts2d2 = torch.stack([torch.from_numpy(xx.flatten() + flowpred[:, :, 0].flatten()), torch.from_numpy(yy.flatten() + flowpred[:, :, 1].flatten()), torch.ones(int(w * h))], dim=0).contiguous().float()

        # Read Pose gt
        gtposes_sourse = readlines(os.path.join(odomgtroot, "{}.txt".format(str(seqmap[seq[0:21]]['mapid']).zfill(2))))
        gtposes_str = [gtposes_sourse[frameidx - int(seqmap[seq[0:21]]['stid'])],
                       gtposes_sourse[frameidx + 1 - int(seqmap[seq[0:21]]['stid'])]]
        gtposes = list()
        for gtposestr in gtposes_str:
            gtpose = np.eye(4).flatten()
            for numstridx, numstr in enumerate(gtposestr.split(' ')):
                gtpose[numstridx] = float(numstr)
            gtpose = np.reshape(gtpose, [4, 4])
            gtposes.append(gtpose)
        posegt = np.linalg.inv(gtposes[1]) @ gtposes[0]

        semanticspred = Image.open(os.path.join(semanticroot, day, seq, 'semantic_prediction/image_02', imgname))
        semanticspred = semanticspred.resize(rgb.size, Image.NEAREST)
        semanticspred = np.array(semanticspred)
        semantic_selector = np.ones(rgb.size[::-1])
        for ll in np.unique(semanticspred).tolist():
            if ll in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]:
                semantic_selector[semanticspred == ll] = 0

        semantic_selector_arr = copy.deepcopy(semantic_selector)
        semantic_selector = (torch.from_numpy(semantic_selector.flatten()) == 1)

        depth_selector = torch.from_numpy((depthmap.numpy() > 0).flatten()) == 1
        pts2d1_gt, pts2d2_gt = acquire_gt_flow(intrinsic, depthmap, posegt, semantic_selector_arr)

        optimization_selector = semantic_selector * depth_selector
        pts2d1sel = torch.stack([pts2d1[0, semantic_selector], pts2d1[1, semantic_selector], pts2d1[2, semantic_selector]], dim=0)
        pts2d2sel = torch.stack([pts2d2[0, semantic_selector], pts2d2[1, semantic_selector], pts2d2[2, semantic_selector]], dim=0)

        depthgt = depthmapf[optimization_selector]
        # eppcocluer.newton_gauss_F(pts2d1_gt.cuda(), pts2d2_gt.cuda(), intrinsic.cuda(), torch.from_numpy(posegt).float().cuda(), depthgt.cuda())
        # eppcocluer.newton_gauss_F(pts2d1_gt, pts2d2_gt, intrinsic, torch.from_numpy(posegt).float(), depthgt)
        eppcocluer.newton_gauss_F(pts2d1sel.cuda(), pts2d2sel.cuda(), intrinsic.cuda(), torch.from_numpy(posegt).float().cuda(), depthgt.cuda())
