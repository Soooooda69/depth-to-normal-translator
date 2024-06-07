import struct

import cv2
import numpy as np


def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras = {"width": width, "height": height, "params": params}
                break
    return cameras

def get_cam_params(calib_path):
    with open(calib_path, 'r') as f:
        data = f.read()
        params = list(map(float, (data.split())))[:-1]
        
    return params


def get_normal_gt(normal_path):
    # retVal: [-1,1]
    normal_gt = cv2.imread(normal_path, -1)
    normal_gt = normal_gt[:, :, ::-1]
    normal_gt = 1 - normal_gt / 65535 * 2
    return normal_gt


def get_depth(depth_path, height, width):
    # with open(depth_path, 'rb') as f:
    #     data_raw = struct.unpack('f' * width * height, f.read(4 * width * height))
    #     z = np.array(data_raw).reshape(height, width)
    z = np.load(depth_path).reshape(height, width)
    # create mask, 1 for foreground, 0 for background
    mask = np.ones_like(z)
    mask[z == 1] = 0

    return z, mask


def vector_normalization(normal, eps=1e-8):
    mag = np.linalg.norm(normal, axis=2)
    normal /= (np.expand_dims(mag, axis=2) + eps)
    return normal


def visualization_map_creation(normal, mask):
    mask = np.expand_dims(mask, axis=2)
    vis = normal * mask + mask - 1
    vis = (1 - vis) / 2  # transform the interval from [-1, 1] to [0, 1]
    return vis


def angle_normalization(err_map):
    err_map[err_map > np.pi / 2] = np.pi - err_map[err_map > np.pi / 2]
    return err_map


def evaluation(n_gt, n_est, mask):
    scale = np.pi / 180
    error_map = np.arccos(np.sum(n_gt * n_est, axis=2))
    error_map = angle_normalization(error_map) / scale
    error_map *= mask
    ea = error_map.sum() / mask.sum()
    return error_map, ea

# def softmax(x):
#     x_exp = np.exp(x)
#     x_sum = np.sum(x_exp)
#     return x_exp / x_sum
#
#
# def softmin(x):
#     x_exp = np.exp(-x)
#     x_sum = np.sum(x_exp)
#     return x_exp / x_sum
