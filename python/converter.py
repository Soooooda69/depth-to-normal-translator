from copy import copy
from typing import Any
import matplotlib.pyplot as plt
from utils import *
from natsort import natsorted
import os 
import argparse
from tqdm import tqdm

class d2nt:
    def __init__(self, root_path: str) -> None:
        self.normal_path = os.path.join(root_path, 'normal')
        os.makedirs(self.normal_path, exist_ok=True)
        self.depth_path = os.path.join(root_path, 'depth')
        self.calib_path = os.path.join(root_path, 'colmap_data/sparse/0/cameras.txt')
        # self.cam_fx, self.cam_fy, self.u0, self.v0 = get_cam_params(self.calib_path)
        self.cam_fx, self.cam_fy, self.u0, self.v0 = read_intrinsics_text(self.calib_path)['params']
        self.h, self.w = read_intrinsics_text(self.calib_path)['height'], read_intrinsics_text(self.calib_path)['width']
        self.depth_files = natsorted([os.path.join(self.depth_path, f) for f in os.listdir(self.depth_path) if f.endswith('.npy')])
        
    def d2n(self) -> Any:
        for depth_file in tqdm(self.depth_files):
            file_num = depth_file.split('/')[-1].split('.')[0]
            depth, mask = get_depth(depth_file, self.h, self.w)
            u_map = np.ones((self.h, 1)) * np.arange(1, self.w + 1) - self.u0
            v_map = np.arange(1, self.h + 1).reshape(self.h, 1) * np.ones((1, self.w)) - self.v0

            Gu, Gv = get_DAG_filter(depth)
            # Depth to Normal Translation
            est_nx = Gu * self.cam_fx
            est_ny = Gv * self.cam_fy
            est_nz = -(depth + v_map * Gv + u_map * Gu)
            est_normal = cv2.merge((est_nx, est_ny, est_nz))
            # vector normalization
            est_normal = vector_normalization(est_normal)
            est_normal = MRF_optim(depth, est_normal)
            np.save(os.path.join(self.normal_path, f'{file_num}.npy'), est_normal[...,None].transpose(3,2,0,1))
            # break
            # visualize the computed normal
            n_vis = visualization_map_creation(est_normal, mask)
            plt.imsave(os.path.join(self.normal_path, f'{file_num}.png'), n_vis)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth to Normal Translator')
    parser.add_argument('--root_path', type=str, help='Path to save the normal maps')
    # parser.add_argument('--normal_path', type=str, help='Path to save the normal maps')
    # parser.add_argument('--depth_path', type=str, help='Path to the depth maps')
    # parser.add_argument('--calib_path', type=str, help='Path to the camera calibration file')
    
    args = parser.parse_args()
    
    convertor = d2nt(args.root_path)
    convertor.d2n()
