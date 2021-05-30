import glob
import os
import PIL.Image as Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from core.utils.frame_utils import readFlowKITTI
from scipy import interpolate
import pickle
import scipy.io

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

def bresenham(x0, y0, x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

def get_intrinsic(calibpath):
    cam2cam = read_calib_file(calibpath)
    K = np.eye(4, dtype=np.float32)
    K[0:3, :] = cam2cam['P_rect_02'].reshape(3, 4)
    return K

train_split_root = '/home/shengjie/Documents/supporting_projects/RAFT/exp_kitti_eigen/splits/train_files.txt'
train_entries = readlines(train_split_root)
kitti_root = '/media/shengjie/disk1/data/Kitti'
optical_flow_root = '/media/shengjie/disk1/Prediction/kittieigen_RAFT_pred'
vls_root = '/media/shengjie/disk1/visualization/quiver_vls'
pose_root = '/media/shengjie/disk1/Prediction/pose_selfdecoders_nosoftmax_photoloss_selected/epoch_006'
os.makedirs(vls_root, exist_ok=True)
sift = cv2.SIFT_create()
for entry in train_entries:
    # entry = '2011_10_03/2011_10_03_drive_0042_sync 0000000763 l'
    # entry = '2011_09_26/2011_09_26_drive_0091_sync 0000000262 l'
    entry = '2011_10_03/2011_10_03_drive_0042_sync 0000000948 l'
    seq, frmidx, _ = entry.split(' ')
    img1_path = os.path.join(kitti_root, seq, 'image_02/data', str(frmidx).zfill(10) + '.png')
    img2_path = os.path.join(kitti_root, seq, 'image_02/data', str(int(frmidx) + 1).zfill(10) + '.png')
    pose_path = os.path.join(pose_root, seq, 'image_02', str(frmidx).zfill(10) + '.pickle')
    if not os.path.exists(img2_path):
        continue

    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    flow_pred, _ = readFlowKITTI(os.path.join(optical_flow_root, seq, 'image_02',  str(frmidx).zfill(10) + '.png'))
    h, w ,_ = flow_pred.shape

    with open(pose_path, 'rb') as f:
        pose = pickle.load(f)
    pose = pose[0]
    R = pose[0:3, 0:3]
    t = pose[0:3, 3]

    intrinsic = get_intrinsic(os.path.join(kitti_root, seq.split('/')[0], 'calib_cam_to_cam.txt'))
    intrinsic = intrinsic[0:3, 0:3]

    epipole1 = intrinsic @ (-R.T @ t)
    epipole1 = epipole1 / epipole1[2]
    epipole2 = intrinsic @ t
    epipole2 = epipole2 / epipole2[2]

    ptsx1 = 417
    ptsy1 = 343
    ptsx2 = 802
    ptsy2 = 343

    sample_density = 10
    pts_line1 = list()
    for pts in bresenham(ptsx1, ptsy1, int(epipole1[0]), int(epipole1[1])):
        pts_line1.append(pts)
    pts_line1 = np.array(pts_line1)
    pts_line1 = pts_line1[0:-5:sample_density]
    pts_line1_flow = flow_pred[pts_line1[:, 1], pts_line1[:, 0]]

    pts_line2 = list()
    for pts in bresenham(ptsx2, ptsy2, int(epipole1[0]), int(epipole1[1])):
        pts_line2.append(pts)
    pts_line2 = np.array(pts_line2)
    pts_line2 = pts_line2[0:-5:sample_density]
    pts_line2_flow = flow_pred[pts_line2[:, 1], pts_line2[:, 0]]

    img1_cv2 = cv2.imread(img1_path)
    img1_cv2 = cv2.cvtColor(img1_cv2, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(img1_cv2, None)
    kps = list()
    for k in kp:
        kps.append(k.pt)
    kps = np.array(kps)
    kps = np.round(kps).astype(np.int)
    selector = (kps[:, 0] >= 0) * (kps[:, 0] < w) * (kps[:, 1] >= 0) * (kps[:, 1] < h)
    kps = kps[selector, :]
    flow = flow_pred[kps[:, 1], kps[:, 0], :]

    scipy.io.savemat('/home/shengjie/Documents/supporting_projects/RAFT/paper_figure_tools/fig.mat',
                     mdict={'kps': kps, 'pts_line1': pts_line1, 'pts_line1_flow':pts_line1_flow, 'flow_kps':flow,
                            'epipole1':epipole1, 'epipole2':epipole2, 'pts_line2':pts_line2, 'pts_line2_flow': pts_line2_flow})

    sv_fold = os.path.join(vls_root, )

    plt.figure(figsize=(16, 9))
    # plt.scatter(kps[:, 0], kps[:, 1], 0.5)
    plt.scatter(epipole1[0], epipole1[1], 30, 'r')
    plt.scatter(epipole2[0], epipole2[1], 30, 'g')
    plt.scatter(pts_line1[:, 0], pts_line1[:, 1], 3, 'b')
    plt.scatter(pts_line1[:, 0] + pts_line1_flow[:, 0], pts_line1[:, 1] + pts_line1_flow[:, 1], 3, 'r')
    # plt.quiver(pts_line1[:, 0], pts_line1[:, 1], pts_line1_flow[:, 0], pts_line1_flow[:, 1], color='y', linewidths=10)
    plt.quiver(kps[:, 0], kps[:, 1], flow[:, 0], flow[:, 1], color='y',  units='width')
    plt.imshow(img1)
    plt.show()

    plt.savefig(os.path.join(vls_root, "{}_{}.png".format(seq.split('/')[1], str(frmidx).zfill(10))))
    plt.close()

