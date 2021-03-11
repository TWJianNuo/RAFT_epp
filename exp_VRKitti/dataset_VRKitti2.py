from __future__ import print_function, division
import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

import random
import numpy as np
import cv2
import PIL.Image as Image

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

from core.utils.frame_utils import read_gen, readFlowVRKitti
import torch.nn.functional as F
from torchvision.transforms import ColorJitter
import copy

def read_splits(project_rootdir):
    split_root = os.path.join(project_rootdir, 'exp_VRKitti', 'splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'training_split.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'evaluation_split.txt'), 'r')]
    return train_entries, evaluation_entries

def read_move_info(intrinsic_path, extrinisic_path, objpose_path, scene_name):
    intrinsic_entries = [x.rstrip('\n') for x in open(intrinsic_path, 'r')]
    intrinsic_dict = dict()
    for entry in intrinsic_entries[1::]:
        horarr = np.array(list(map(float, entry.split(' '))))
        frmidx = int(horarr[0])
        camidx = int(horarr[1])
        intrinsic_key = "{}_frm{}_cam{}".format(scene_name, str(frmidx).zfill(5), camidx)

        intrinsic = np.eye(4)
        intrinsic[0, 0] = horarr[2]
        intrinsic[1, 1] = horarr[3]
        intrinsic[0, 2] = horarr[4]
        intrinsic[1, 2] = horarr[5]
        intrinsic_dict[intrinsic_key] = intrinsic

    extrisnic_entries = [x.rstrip('\n') for x in open(extrinisic_path, 'r')]
    extrinsic_dict = dict()
    for entry in extrisnic_entries[1::]:
        horarr = np.array(list(map(float, entry.split(' '))))
        frmidx = int(horarr[0])
        camidx = int(horarr[1])
        extrinsic_key = "{}_frm{}_cam{}".format(scene_name, str(frmidx).zfill(5), camidx)
        extrinsic_dict[extrinsic_key] = np.reshape(np.array([horarr[2::]]), [4,4])

    objpose_entries = [x.rstrip('\n') for x in open(objpose_path, 'r')]
    objpose_dict = dict()
    for entry in objpose_entries[1::]:
        horarr = np.array(list(map(float, entry.split(' '))))
        frmidx = int(horarr[0])
        camidx = int(horarr[1])
        trackidx = int(horarr[2])
        objpose_key = "{}_frm{}_cam{}_obj{}".format(scene_name, str(frmidx).zfill(5), camidx, str(trackidx).zfill(5))

        objpose_dict[objpose_key] = np.array(horarr[3::])

    return intrinsic_dict, extrinsic_dict, objpose_dict

class VirtualKITTI2(data.Dataset):
    def __init__(self, args, inheight, inwidth, isapproxpose, root='datasets/KITTI', entries=None, istrain=True):
        super(data.Dataset, self).__init__()
        self.args = args
        self.root = root
        self.entries = entries
        self.istrain = istrain
        self.isapproxpose = isapproxpose

        self.inheight = inheight
        self.inwidth = inwidth

        self.intrinsic_dict = dict()
        self.extrinsic_dict = dict()
        self.objpose_dict = dict()

        self.rgb1_paths = list()
        self.rgb2_paths = list()
        self.flow_paths = list()
        self.depth_paths = list()
        self.ins_paths = list()

        self.register_info()
        self.minptsnum = 100
        self.maxinsnum = self.args.maxinsnum

        featurew = int(self.inwidth / 4)
        featureh = int(self.inheight / 4)
        xx, yy = np.meshgrid(range(featurew), range(featureh), indexing='xy')
        xx = (xx / featurew - 0.5) * 2
        yy = (yy / featureh - 0.5) * 2
        self.sample_pts = torch.stack([torch.from_numpy(xx).float(), torch.from_numpy(yy).float()], dim=-1)

        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25/3.14)
        self.asymmetric_color_aug_prob = 0.2

    def register_info(self):
        registered_scene = list()
        for entry in self.entries:
            sceneidx, envn, frmidx = entry.split(' ')

            frmidx = int(frmidx)

            rgb1path = os.path.join(self.root, "Scene{}".format(sceneidx.zfill(2)), envn, 'frames', 'rgb', 'Camera_0', "rgb_{}.jpg".format(str(frmidx).zfill(5)))
            rgb2path = os.path.join(self.root, "Scene{}".format(sceneidx.zfill(2)), envn, 'frames', 'rgb', 'Camera_0', "rgb_{}.jpg".format(str(frmidx + 1).zfill(5)))
            depthpath = os.path.join(self.root, "Scene{}".format(sceneidx.zfill(2)), envn, 'frames', 'depth', 'Camera_0', "depth_{}.png".format(str(frmidx).zfill(5)))
            flowpath = os.path.join(self.root, "Scene{}".format(sceneidx.zfill(2)), envn, 'frames', 'forwardFlow', 'Camera_0', "flow_{}.png".format(str(frmidx).zfill(5)))
            insmappath = os.path.join(self.root, "Scene{}".format(sceneidx.zfill(2)), envn, 'frames', 'instanceSegmentation', 'Camera_0', "instancegt_{}.png".format(str(frmidx).zfill(5)))

            self.rgb1_paths.append(rgb1path)
            self.rgb2_paths.append(rgb2path)
            self.depth_paths.append(depthpath)
            self.flow_paths.append(flowpath)
            self.ins_paths.append(insmappath)

            extrinisic_path = os.path.join(self.root, "Scene{}".format(sceneidx.zfill(2)), envn, 'extrinsic.txt')
            intrinsic_path = os.path.join(self.root, "Scene{}".format(sceneidx.zfill(2)), envn, 'intrinsic.txt')
            objpose_path = os.path.join(self.root, "Scene{}".format(sceneidx.zfill(2)), envn, 'pose.txt')
            scene_name = "Scene{}_{}".format(sceneidx.zfill(2), envn)
            if scene_name not in registered_scene:
                intrinsic_dict, extrinsic_dict, objpose_dict = read_move_info(intrinsic_path, extrinisic_path, objpose_path, scene_name)
                self.intrinsic_dict.update(intrinsic_dict)
                self.extrinsic_dict.update(extrinsic_dict)
                self.objpose_dict.update(objpose_dict)
                registered_scene.append(scene_name)

    def get_tag(self, index):
        entry = self.entries[index]
        sceneidx, weather, frmidx = entry.split(' ')
        tag = "scene{}_{}_{}".format(sceneidx.zfill(2), weather, frmidx.zfill(5))
        return tag

    def crop_img(self, img, left, top, crph, crpw):
        img_cropped = img[top:top+crph, left:left+crpw]
        return img_cropped

    def aug_crop(self, img1, img2, flowmap, depthmap, instancemap, intrinsic):
        if img1.ndim == 3:
            h, w, _ = img1.shape
        else:
            h, w = img1.shape

        crph = self.inheight
        crpw = self.inwidth

        if self.istrain:
            left = np.random.randint(0, w - crpw - 1, 1).item()
        else:
            left = int((w - crpw) / 2)
        top = int(h - crph)

        intrinsic[0, 2] -= left
        intrinsic[1, 2] -= top

        img1 = self.crop_img(img1, left=left, top=top, crph=crph, crpw=crpw)
        img2 = self.crop_img(img2, left=left, top=top, crph=crph, crpw=crpw)
        flowmap = self.crop_img(flowmap, left=left, top=top, crph=crph, crpw=crpw)
        depthmap = self.crop_img(depthmap, left=left, top=top, crph=crph, crpw=crpw)
        instancemap = self.crop_img(instancemap, left=left, top=top, crph=crph, crpw=crpw)

        return img1, img2, flowmap, depthmap, instancemap, intrinsic

    def colorjitter(self, img1, img2):
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def __getitem__(self, index):
        while True:
            try:
                img1, img2, flowmap, depthmap, instancemap, intrinsic = self.read_imgpair_flow_depth_instance_intrinsic(index)
                img1, img2, flowmap, depthmap, instancemap, intrinsic = self.aug_crop(img1, img2, flowmap, depthmap, instancemap, intrinsic)
                break
            except:
                index = index + 1

        if self.istrain:
            img1, img2 = self.colorjitter(img1, img2)

        relpose, obj_poses, obj_approxmv, obj_approxposes = self.get_mvinfo(index, instancemap)
        # self.validate_mvinfo(img1, img2, depthmap, instancemap, intrinsic, relpose, obj_poses, obj_approxposes, flowmap, index)

        if self.isapproxpose:
            renamed_ins, renamed_poses, renamed_ang, renamed_scale = self.rename_instancemap(instancemap, flowmap, relpose, obj_approxposes, obj_approxmv)
        else:
            renamed_ins, renamed_poses, renamed_ang, renamed_scale = self.rename_instancemap(instancemap, flowmap, relpose, obj_poses, obj_approxmv)

        renamed_ins_featuresize = F.grid_sample(torch.from_numpy(renamed_ins).float().view([1, 1, self.inheight, self.inwidth]), self.sample_pts.unsqueeze(0), mode='nearest', align_corners=True)
        renamed_ins_featuresize = renamed_ins_featuresize.squeeze().int().numpy()

        intrinsic_resize = np.eye(4)
        resizeM = np.eye(3)
        resizeM[0, 0] = 0.25
        resizeM[1, 1] = 0.25
        intrinsic_resize[0:3, 0:3] = resizeM @ intrinsic[0:3, 0:3]
        # self.validate_rename(img1, img2, depthmap, renamed_ins, intrinsic, renamed_poses, flowmap)

        tag = self.get_tag(index)
        data_blob = self.wrapup(img1, img2, flowmap, depthmap, intrinsic, intrinsic_resize, renamed_ins, renamed_ins_featuresize, renamed_poses, renamed_ang, renamed_scale, tag)
        return data_blob

    def __len__(self):
        return len(self.entries)

    def wrapup(self, img1, img2, flowmap, depthmap, intrinsic, intrinsic_resize, renamed_ins, renamed_ins_featuresize, renamed_poses, renamed_ang, renamed_scale, tag):
        img1 = torch.from_numpy(img1).permute([2, 0, 1]).float()
        img2 = torch.from_numpy(img2).permute([2, 0, 1]).float()
        flowmap = torch.from_numpy(flowmap).permute([2, 0, 1]).float()
        depthmap = torch.from_numpy(depthmap).unsqueeze(0).float()
        intrinsic = torch.from_numpy(intrinsic).float()
        intrinsic_resize = torch.from_numpy(intrinsic_resize).float()
        renamed_ins = torch.from_numpy(renamed_ins).unsqueeze(0).int()
        renamed_ins_featuresize = torch.from_numpy(renamed_ins_featuresize).unsqueeze(0).int()
        renamed_poses_pad = torch.zeros([self.maxinsnum, 4, 4], dtype=torch.float)
        renamed_poses_pad[0:renamed_poses.shape[0], :, :] = torch.from_numpy(renamed_poses).float()
        renamed_ang_pad = torch.zeros([self.maxinsnum, 1, 1], dtype=torch.float)
        renamed_ang_pad[0:renamed_ang.shape[0], :, :] = torch.from_numpy(renamed_ang).float()
        renamed_scale_pad = torch.zeros([self.maxinsnum, 1, 1], dtype=torch.float)
        renamed_scale_pad[0:renamed_scale.shape[0], :, :] = torch.from_numpy(renamed_scale).float()

        data_blob = dict()
        data_blob['img1'] = img1
        data_blob['img2'] = img2
        data_blob['flowmap'] = flowmap
        data_blob['depthmap'] = depthmap
        data_blob['intrinsic'] = intrinsic
        data_blob['intrinsic_resize'] = intrinsic_resize
        data_blob['insmap'] = renamed_ins
        data_blob['insmap_featuresize'] = renamed_ins_featuresize
        data_blob['poses'] = renamed_poses_pad
        data_blob['ang'] = renamed_ang_pad
        data_blob['scale'] = renamed_scale_pad
        data_blob['tag'] = tag

        return data_blob

    def rename_instancemap(self, instancemap, flowmap, relpose, obj_approxposes, obj_approxmv):
        invalid = flowmap[:, :, 0] == 0

        # Reinit a rigid obj instancemap
        renamed_ins = np.zeros_like(instancemap)
        renamed_ins[instancemap == -1] = 0
        renamed_ins[invalid] = -1

        # Background is the first rigid obj
        inscounts = 1

        uniqueidx = list()
        for k in np.unique(instancemap):
            if k == -1:
                continue
            if np.sum(instancemap == k) > self.minptsnum and k in obj_approxposes.keys() and inscounts < self.maxinsnum:
                uniqueidx.append(k)
                renamed_ins[instancemap == k] = inscounts
                inscounts += 1
            else:
                renamed_ins[instancemap == k] = -1

        renamed_poses = np.zeros([len(uniqueidx) + 1, 4, 4])
        for k in range(len(uniqueidx) + 1):
            if k == 0:
                renamed_poses[k] = relpose
            else:
                renamed_poses[k] = obj_approxposes[uniqueidx[k-1]]

        renamed_ang = np.zeros([len(uniqueidx) + 1, 1, 1])
        renamed_scale = np.zeros([len(uniqueidx) + 1, 1, 1])
        for k in range(len(uniqueidx) + 1):
            if k == 0:
                continue
            else:
                renamed_ang[k, 0, 0] = obj_approxmv["obj{}_ang".format(str(uniqueidx[k-1]).zfill(2))]
                renamed_scale[k, 0, 0] = obj_approxmv["obj{}_scale".format(str(uniqueidx[k - 1]).zfill(2))]
        return renamed_ins, renamed_poses, renamed_ang, renamed_scale

    def ang2R(self, ang):
        # Order is roll - pitch - yaw
        R_x = np.array([[1,                 0,                  0],
                        [0,                 np.cos(ang[0]),     -np.sin(ang[0])],
                        [0,                 np.sin(ang[0]),     np.cos(ang[0])]])

        R_y = np.array([[np.cos(ang[1]),    0,                  np.sin(ang[1])],
                        [0,                 1,                  0],
                        [-np.sin(ang[1]),   0,                  np.cos(ang[1])]])

        R_z = np.array([[np.cos(ang[2]),    -np.sin(ang[2]),    0],
                        [np.sin(ang[2]),    np.cos(ang[2]),     0],
                        [0,                 0,                  1]])

        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def R2ang(self, R):
        # This is not an efficient implementation
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        ang0 = np.arctan2(R[2, 1], R[2, 2])
        ang1 = np.arctan2(-R[2, 0], sy)
        ang2 = np.arctan2(R[1, 0], R[0, 0])
        ang = np.array([ang0, ang1, ang2])
        return ang

    def get_mvinfo(self, index, instancemap):
        sceneidx, envn, frmidx = self.entries[index].split(' ')
        frmidx = int(frmidx)
        camidx = int(0)
        scene_name = "Scene{}_{}".format(sceneidx.zfill(2), envn)

        extrinsic_key1 = "{}_frm{}_cam{}".format(scene_name, str(frmidx).zfill(5), camidx)
        extrinsic_key2 = "{}_frm{}_cam{}".format(scene_name, str(frmidx + 1).zfill(5), camidx)

        exrtinsic1 = self.extrinsic_dict[extrinsic_key1]
        exrtinsic2 = self.extrinsic_dict[extrinsic_key2]

        relpose = exrtinsic2 @ np.linalg.inv(exrtinsic1)

        obj_poses = dict()
        obj_approxmv = dict()
        obj_approxposes = dict()
        for k in np.unique(instancemap):
            if k == -1:
                continue
            objpose_key1 = "{}_frm{}_cam{}_obj{}".format(scene_name, str(frmidx).zfill(5), camidx, str(k).zfill(5))
            objpose_key2 = "{}_frm{}_cam{}_obj{}".format(scene_name, str(frmidx + 1).zfill(5), camidx, str(k).zfill(5))
            if objpose_key1 in self.objpose_dict and objpose_key2 in self.objpose_dict:
                obj_trans_ang1 = self.objpose_dict[objpose_key1][-12:-6:]
                obj_trans_ang2 = self.objpose_dict[objpose_key2][-12:-6:]

                objpose1 = np.eye(4)
                objpose1[0:3, 3] = obj_trans_ang1[0:3]
                angs1 = np.zeros([3])
                angs1[1] = obj_trans_ang1[3]
                objpose1[0:3, 0:3] = self.ang2R(angs1)

                objpose2 = np.eye(4)
                objpose2[0:3, 3] = obj_trans_ang2[0:3]
                angs2 = np.zeros([3])
                angs2[1] = obj_trans_ang2[3]
                objpose2[0:3, 0:3] = self.ang2R(angs2)

                objpose = exrtinsic2 @ objpose2 @ np.linalg.inv(exrtinsic1 @ objpose1)
                objpose_absolute = np.linalg.inv(relpose) @ objpose

                objmv_absolute = objpose_absolute[0:3, 3]
                objmv_absolute[1] = 0

                objdirection = objmv_absolute / np.sqrt(np.sum(objmv_absolute ** 2 + 1e-10))
                objdirection = np.arctan2(-objdirection[2], objdirection[0])

                ang = objdirection
                scale = np.sqrt(np.sum(objmv_absolute ** 2))
                obj_approxmv["obj{}_ang".format(str(k).zfill(2))] = ang
                obj_approxmv["obj{}_scale".format(str(k).zfill(2))] = scale
                obj_approxposes[k] = self.approx_objrelpose(ang, scale, relpose)
                # obj_pos1_cam1 = exrtinsic1 @ objpose1 @ np.array([[0, 0, 0, 1]]).T
                # obj_pos2_cam1 = exrtinsic1 @ objpose2 @ np.array([[0, 0, 0, 1]]).T
                # diff = objpose_absolute @ obj_pos1_cam1 - obj_pos2_cam1

                obj_poses[k] = objpose

        return relpose, obj_poses, obj_approxmv, obj_approxposes

    def approx_objmv(self, ang, scale):
        M = np.eye(3)
        M[0, 0] = np.cos(ang)
        M[0, 2] = np.sin(ang)
        M[2, 0] = -np.sin(ang)
        M[2, 2] = np.cos(ang)

        mv = M @ np.array([[1, 0, 0]]).T * scale
        return mv[:, 0]

    def approx_objrelpose(self, ang, scale, relpose):
        M = np.eye(3)
        M[0, 0] = np.cos(ang)
        M[0, 2] = np.sin(ang)
        M[2, 0] = -np.sin(ang)
        M[2, 2] = np.cos(ang)

        mv = M @ np.array([[1, 0, 0]]).T * scale
        mv = mv[:, 0]

        approx_objpose_absolute = np.eye(4)
        approx_objpose_absolute[0:3, 3] = mv
        approx_objpose_absolute = approx_objpose_absolute @ relpose
        return approx_objpose_absolute

    def validate_mvinfo(self, img1, img2, depthmap, instancemap, intrinsic, relpose, obj_poses, obj_approxposes, flowmap, index):
        h, w = depthmap.shape
        import matplotlib.pyplot as plt
        cm = plt.get_cmap('magma')
        vmax = 0.15

        staticsel = instancemap == -1
        samplenume = 10000
        rndx = np.random.randint(0, w, [samplenume])
        rndy = np.random.randint(0, h, [samplenume])
        rndsel = staticsel[rndy, rndx]

        rndx = rndx[rndsel]
        rndy = rndy[rndsel]
        rndd = depthmap[rndy, rndx]

        rndpts = np.stack([rndx * rndd, rndy * rndd, rndd, np.ones_like(rndd)], axis=0)
        rndpts = intrinsic @ relpose @ np.linalg.inv(intrinsic) @ rndpts
        rndptsx = rndpts[0, :] / rndpts[2, :]
        rndptsy = rndpts[1, :] / rndpts[2, :]

        rndflowx = flowmap[rndy, rndx, 0]
        rndflowy = flowmap[rndy, rndx, 1]

        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        objsample = 1000
        orgpts = list()
        flowpts = list()
        objposepts = list()
        objposepts_approx = list()
        colors = list()
        for k in obj_poses.keys():
            obj_pose = obj_poses[k]
            obj_pose_approx = obj_approxposes[k]
            obj_sel = instancemap == k

            if np.sum(obj_sel) < objsample:
                continue

            rndidx = np.random.randint(0, np.sum(obj_sel), objsample)

            objxxf = xx[obj_sel][rndidx]
            objyyf = yy[obj_sel][rndidx]

            objxxff = flowmap[objyyf, objxxf, 0]
            objyyff = flowmap[objyyf, objxxf, 1]
            objdf = depthmap[objyyf, objxxf]
            objcolor = 1 / objdf / vmax
            objcolor = cm(objcolor)

            objpts3d = np.stack([objxxf * objdf, objyyf * objdf, objdf, np.ones_like(objdf)], axis=0)
            objpts3d = intrinsic @ obj_pose @ np.linalg.inv(intrinsic) @ objpts3d
            objpts3dx = objpts3d[0, :] / objpts3d[2, :]
            objpts3dy = objpts3d[1, :] / objpts3d[2, :]

            objpts3d = np.stack([objxxf * objdf, objyyf * objdf, objdf, np.ones_like(objdf)], axis=0)
            objpts3d = intrinsic @ obj_pose_approx @ np.linalg.inv(intrinsic) @ objpts3d
            objpts3dx_approx = objpts3d[0, :] / objpts3d[2, :]
            objpts3dy_approx = objpts3d[1, :] / objpts3d[2, :]

            objxxf_o = objxxf + objxxff
            objyyf_o = objyyf + objyyff

            orgpts.append(np.stack([objxxf, objyyf], axis=0))
            flowpts.append(np.stack([objxxf_o, objyyf_o], axis=0))
            objposepts.append(np.stack([objpts3dx, objpts3dy], axis=0))
            objposepts_approx.append(np.stack([objpts3dx_approx, objpts3dy_approx], axis=0))
            colors.append(objcolor)

        tnp = 1 / rndd / vmax
        tnp = cm(tnp)

        fig = plt.figure(figsize=(16, 9))
        fig.add_subplot(4, 1, 1)
        plt.scatter(rndx, rndy, 1, tnp)
        for k in range(len(orgpts)):
            plt.scatter(orgpts[k][0], orgpts[k][1], 1, colors[k])
        plt.imshow(img1)

        fig.add_subplot(4, 1, 2)
        plt.scatter(rndptsx, rndptsy, 1, tnp)
        for k in range(len(objposepts)):
            plt.scatter(objposepts[k][0], objposepts[k][1], 1, colors[k])
        plt.imshow(img2)

        fig.add_subplot(4, 1, 3)
        plt.scatter(rndptsx, rndptsy, 1, tnp)
        for k in range(len(objposepts_approx)):
            plt.scatter(objposepts_approx[k][0], objposepts_approx[k][1], 1, colors[k])
        plt.imshow(img2)

        fig.add_subplot(4, 1, 4)
        plt.scatter(rndx + rndflowx, rndy + rndflowy, 1, tnp)
        for k in range(len(flowpts)):
            plt.scatter(flowpts[k][0], flowpts[k][1], 1, colors[k])
        plt.imshow(img2)

        seqidx, weather, frmidx = self.entries[index].split(' ')
        plt.savefig(os.path.join('/media/shengjie/disk1/visualization/2021_03/vrkitti_approx_pose', "{}_{}_{}".format(seqidx, weather, frmidx)))
        plt.close()
        return

    def validate_rename(self, img1, img2, depthmap, renamed_ins, intrinsic, renamed_poses, flowmap):
        h, w = depthmap.shape
        import matplotlib.pyplot as plt
        cm = plt.get_cmap('magma')
        vmax = 0.15
        intrinsic33 = intrinsic[0:3, 0:3]

        def t2T(t):
            T = np.zeros([3, 3])
            T[0, 1] = -t[2]
            T[0, 2] = t[1]
            T[1, 0] = t[2]
            T[1, 2] = -t[0]
            T[2, 0] = -t[1]
            T[2, 1] = t[0]
            return T

        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        objsample = 1000
        orgpts = list()
        flowpts = list()
        objposepts = list()
        colors = list()
        for k in np.unique(renamed_ins):
            if k == -1:
                continue
            obj_pose = renamed_poses[k]
            obj_sel = renamed_ins == k

            if np.sum(obj_sel) < objsample:
                continue

            rndidx = np.random.randint(0, np.sum(obj_sel), objsample)

            objxxf = xx[obj_sel][rndidx]
            objyyf = yy[obj_sel][rndidx]

            objxxff = flowmap[objyyf, objxxf, 0]
            objyyff = flowmap[objyyf, objxxf, 1]
            objdf = depthmap[objyyf, objxxf]
            objcolor = 1 / objdf / vmax
            objcolor = cm(objcolor)

            objpts3d = np.stack([objxxf * objdf, objyyf * objdf, objdf, np.ones_like(objdf)], axis=0)
            objpts3d = intrinsic @ obj_pose @ np.linalg.inv(intrinsic) @ objpts3d
            objpts3dx = objpts3d[0, :] / objpts3d[2, :]
            objpts3dy = objpts3d[1, :] / objpts3d[2, :]

            objxxf_o = objxxf + objxxff
            objyyf_o = objyyf + objyyff

            objpts1 = np.stack([objxxf, objyyf, np.ones_like(objxxf)], axis=0)
            objpts2 = np.stack([objxxf_o, objyyf_o, np.ones_like(objpts3dx)], axis=0)

            R = obj_pose[0:3, 0:3]
            t = obj_pose[0:3, 3] / np.sqrt(np.sum(obj_pose[0:3, 3] ** 2))
            T = t2T(t)
            F = T @ R
            E = np.linalg.inv(intrinsic33).T @ F @ np.linalg.inv(intrinsic33)
            cost = np.sum(objpts2.T @ E * objpts1.T, axis=1)
            print("obj %d mean constrain is : %f" % (k, np.abs(cost).mean()))

            orgpts.append(np.stack([objxxf, objyyf], axis=0))
            flowpts.append(np.stack([objxxf_o, objyyf_o], axis=0))
            objposepts.append(np.stack([objpts3dx, objpts3dy], axis=0))
            colors.append(objcolor)

        fig = plt.figure(figsize=(16, 9))
        fig.add_subplot(3, 1, 1)
        for k in range(len(orgpts)):
            plt.scatter(orgpts[k][0], orgpts[k][1], 1, colors[k])
        plt.imshow(img1)

        fig.add_subplot(3, 1, 2)
        for k in range(len(objposepts)):
            plt.scatter(objposepts[k][0], objposepts[k][1], 1, colors[k])
        plt.imshow(img2)

        fig.add_subplot(3, 1, 3)
        for k in range(len(flowpts)):
            plt.scatter(flowpts[k][0], flowpts[k][1], 1, colors[k])
        plt.imshow(img2)
        plt.show()
        return

    def read_imgpair_flow_depth_instance_intrinsic(self, index):
        img1 = np.array(read_gen(self.rgb1_paths[index])).astype(np.uint8)
        img2 = np.array(read_gen(self.rgb2_paths[index])).astype(np.uint8)

        flowmap = np.array(readFlowVRKitti(self.flow_paths[index])).astype(np.float32)

        depthmap = np.array(cv2.imread(self.depth_paths[index], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)).astype(np.float32)
        depthmap = depthmap / 100

        instancemap = np.array(Image.open(self.ins_paths[index]))
        instancemap = np.array(instancemap).astype(np.int32) - 1

        sceneidx, envn, frmidx = self.entries[index].split(' ')
        frmidx = int(frmidx)
        camidx = int(0)
        scene_name = "Scene{}_{}".format(sceneidx.zfill(2), envn)

        intrinsic_key = "{}_frm{}_cam{}".format(scene_name, str(frmidx).zfill(5), camidx)
        intrinsic = copy.deepcopy(self.intrinsic_dict[intrinsic_key])

        return img1, img2, flowmap, depthmap, instancemap, intrinsic

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inheight', type=int, default=192)
    parser.add_argument('--inwidth', type=int, default=1088)
    parser.add_argument('--maxinsnum', type=int, default=20)
    parser.add_argument('--maxscale', type=float, default=10)

    torch.manual_seed(1234)
    np.random.seed(1234)

    args = parser.parse_args()

    vrkittiroot = '/media/shengjie/disk1/data/virtual_kitti_organized'
    project_rootdir = '/home/shengjie/Documents/supporting_projects/RAFT'
    bz = 2
    width = 1242
    height = 375

    train_entries, evaluation_entries = read_splits(project_rootdir)
    vrkitti2 = VirtualKITTI2(root=vrkittiroot, entries=train_entries, args=args)
    vrkitti2loader = DataLoader(vrkitti2, batch_size=bz, pin_memory=False, shuffle=True, num_workers=12, drop_last=True)

    for idx, data_blob in enumerate(vrkitti2loader):
        print(idx)

