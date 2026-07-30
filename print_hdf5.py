import h5py

def print_hdf5_structure(file_path):
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"[Dataset] {name} -> shape: {obj.shape}, dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"[Group]   {name}")
    
    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path}")
        print("Attributes:", dict(f.attrs))  # 查看文件级属性（比如 sim/real）
        f.visititems(visitor)

# 用法：替换成你 ACT 训练数据里的任意一个 episode_0.hdf5 文件路径
#print_hdf5_structure("E:/Adapted_ACT/adapted_act/data/episode_0.hdf5")

import h5py
import numpy as np
import cv2
import os

def export_hdf5_to_folder(hdf5_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(hdf5_path, 'r') as f:
        # 1. 导出动作和关节状态（存为 .npy 或 .txt）
        for key in ['action', 'qpos', 'qvel']:
            if key in f['observations']:
                data = f[f'observations/{key}'][()]
                np.save(os.path.join(output_dir, f'{key}.npy'), data)
            elif key in f:
                data = f[key][()]
                np.save(os.path.join(output_dir, f'{key}.npy'), data)
        
        # 2. 导出图像（按相机分类，存为 PNG）
        if 'observations/images' in f:
            for cam_name in f['observations/images'].keys():
                cam_dir = os.path.join(output_dir, 'images', cam_name)
                os.makedirs(cam_dir, exist_ok=True)
                images = f[f'observations/images/{cam_name}'][()]
                for t in range(images.shape[0]):
                    img = images[t]  # (H, W, C)
                    # 如果图像是 uint8，直接存；如果是 float，先转回 0-255
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(cam_dir, f'frame_{t:04d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # 3. 导出元数据（文件属性）
        with open(os.path.join(output_dir, 'metadata.txt'), 'w') as meta_f:
            for k, v in f.attrs.items():
                meta_f.write(f"{k}: {v}\n")
    
    print(f"Exported to {output_dir}")

# 用法
export_hdf5_to_folder("E:/Adapted_ACT/adapted_act/data/episode_0.hdf5", "./exported_episode")