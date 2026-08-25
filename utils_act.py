import numpy as np
import torch
import os

from torch.utils.data import TensorDataset, DataLoader

import json
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform
from PIL import Image

import IPython
e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names #以cam_id为名
        self.norm_stats = norm_stats
        self.is_sim = True
        self.episode_lens = []
        self.max_episode_len = 0
        self.episode_folds=sorted(os.listdir(os.path.join(self.dataset_dir,'train')))
        #self.__getitem__(0) # initialize self.is_sim
        
        # find the maximum episode len
        for episode_id in episode_ids:
            demo_path=os.path.join(self.dataset_dir,'train',self.episode_folds[episode_id])
            with open(os.path.join(demo_path, "metadata.json"), "r") as f:
                                meta = json.load(f)
            #tcp gripper使用第一个相机的文件夹
            cam_path=os.path.join(demo_path,f"cam_{self.camera_names[0]}")
            master_frame_ids = [
                            int(os.path.splitext(x)[0]) 
                            for x in sorted(os.listdir(os.path.join(demo_path,f"cam_{self.camera_names[0]}", "color"))) 
                            if int(os.path.splitext(x)[0]) <= meta["finish_time"]
                        ]
            episode_len=len(master_frame_ids)
            self.episode_lens.append(episode_len)
            if episode_len > self.max_episode_len:
                self.max_episode_len = episode_len
        

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode
        #目的：每次选取一个demo,抽选合法的strat_ts并得到带掩码的动作序列 位置序列
        episode_id = self.episode_ids[index]
        demo_path=os.path.join(self.dataset_dir,'train',self.episode_folds[episode_id])
        with open(os.path.join(demo_path, "metadata.json"), "r") as f:
                            meta = json.load(f)
        #tcp gripper使用第一个相机的文件夹
        cam_path=os.path.join(demo_path,f"cam_{self.camera_names[0]}")
        
        #get frame ids for each camera
        master_frame_ids = [
                int(os.path.splitext(x)[0]) 
                for x in sorted(os.listdir(os.path.join(demo_path,f"cam_{self.camera_names[0]}", "color"))) 
                if int(os.path.splitext(x)[0]) <= meta["finish_time"]
            ]
        slave_frame_ids = [
                        int(os.path.splitext(x)[0]) 
                        for x in sorted(os.listdir(os.path.join(demo_path,f"cam_{self.camera_names[1]}", "color"))) 
                        if int(os.path.splitext(x)[0]) <= meta["finish_time"]
                    ]
        episode_len=len(master_frame_ids)-1
        
        
        # merge tcp data and gripper data to get action data
        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)
        all_tcp_data=[]
        all_gripper_data=[]
        tcp_path=os.path.join(cam_path,'tcp')
        gripper_path=os.path.join(cam_path,'gripper_command')
        for cur_idx in range(len(master_frame_ids) - 1):
                    tcp=np.load(os.path.join(tcp_path,f'{master_frame_ids[cur_idx]}.npy'))
                    gripper=np.load(os.path.join(gripper_path,f'{master_frame_ids[cur_idx]}.npy'))
                    all_tcp_data.append(tcp[:7])
                    all_gripper_data.append(decode_gripper_width(gripper[0]))
        all_tcp_data=np.stack(all_tcp_data)
        all_gripper_data=np.stack(all_gripper_data)            
            
        # rotation transformation (to 6d)复用RISE
        all_tcp_data=xyz_rot_transform(all_tcp_data, from_rep = "quaternion", to_rep = "rotation_6d")
        all_action_data = np.concatenate((all_tcp_data, all_gripper_data[..., np.newaxis]), axis = -1)# (N, 10)
        
        # get observation at start_ts only
        qpos=all_action_data[start_ts]

        image_dict = dict()
        
        # 对于全局相机，时间戳与图片对应；对于腕部相机，找到当前start ts索引的时间戳最近的时间戳的图像
        master_id=master_frame_ids[start_ts]
        image_dict[self.camera_names[0]] = np.array(Image.open(os.path.join(demo_path,f"cam_{self.camera_names[0]}", "color", f"{master_id}.png"))
        )
        slave_frame_ids_np=np.array(slave_frame_ids)
        counterpart_start_ts = np.argmin(np.abs(slave_frame_ids_np - master_id))
        slave_id=slave_frame_ids[counterpart_start_ts]
        image_dict[self.camera_names[1]] = np.array(Image.open(os.path.join(demo_path,f"cam_{self.camera_names[1]}", "color", f"{slave_id}.png")))
            
        
        # get all actions after and including start_ts
        action = all_action_data[min(start_ts + 1, len(all_action_data)-1):] # hack removed
        action_len =  episode_len - (start_ts + 1) # hack, to make timesteps more aligned
        original_action_shape = all_action_data.shape
        
        # create mask padding
        is_sim = True
        self.is_sim = is_sim
        
        padded_action = np.zeros((self.max_episode_len-1,original_action_shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.max_episode_len-1)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        # norm_stats 是 numpy(float64)，混算会把 float32 张量提升成 float64，
        # 喂给 float32 的 nn.Linear 会错
        action_data = ((action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]).float()
        qpos_data = ((qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]).float()

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes,split,cam_ids):
    #all_qpos_data = []
    #all_action_data = []
    
    all_tcp_data=[]
    all_gripper_data=[]
    
    #目的：遍历所有demo,拿到npy并整合、统计全局统计量
    dataset_path= os.path.join(dataset_dir, split)
    all_demos = sorted(os.listdir(dataset_path))
    num_demos = len(all_demos)
    for i in range(num_demos):
        demo_path=os.path.join(dataset_path,all_demos[i])
        #两个摄像头的tcp gripper数据实际上是同一套 所以只读全局相机的
        cam_id  =cam_ids[0]
        cam_path=os.path.join(demo_path,f'cam_{cam_id}')
        if not os.path.exists(cam_path):
            continue
        tcp_path=os.path.join(cam_path,'tcp')
        gripper_path=os.path.join(cam_path,'gripper_command')
        with open(os.path.join(demo_path, "metadata.json"), "r") as f:
                meta = json.load(f)
        frame_ids = [
                int(os.path.splitext(x)[0]) 
                for x in sorted(os.listdir(os.path.join(cam_path, "color"))) 
                if int(os.path.splitext(x)[0]) <= meta["finish_time"]
            ]
        for cur_idx in range(len(frame_ids) - 1):
            tcp=np.load(os.path.join(tcp_path,f'{frame_ids[cur_idx]}.npy'))
            gripper=np.load(os.path.join(gripper_path,f'{frame_ids[cur_idx]}.npy'))
            all_tcp_data.append(tcp[:7])
            all_gripper_data.append(decode_gripper_width(gripper[0]))
    all_tcp_data=np.stack(all_tcp_data)
    all_gripper_data=np.stack(all_gripper_data)            
    
    # rotation transformation (to 6d)复用RISE
    all_tcp_data=xyz_rot_transform(all_tcp_data, from_rep = "quaternion", to_rep = "rotation_6d")
    all_action_data = np.concatenate((all_tcp_data, all_gripper_data[..., np.newaxis]), axis = -1)# (N, 10)


    # normalize action data
    action_mean = all_action_data.mean(axis=0,keepdims=True)
    action_std = all_action_data.std(axis=0,keepdims=True)
    action_std = np.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    #qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    #qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    #qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
    qpos_mean=action_mean.copy()
    qpos_std=action_std.copy()

    qpos=all_action_data[0]
    
    stats = {"action_mean": action_mean.squeeze(), "action_std": action_std.squeeze(),
             "qpos_mean": qpos_mean.squeeze(), "qpos_std": qpos_std.squeeze(),
             "example_qpos": qpos}
    
    print("action_mean shape:", action_mean.shape)  # (10,)
    print("action_std shape:", action_std.shape)    
    print("example_qpos shape:", qpos.shape) 

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    #由于每个demo是变长的，indices要为每个episode_len独立生成
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    split='train'
    assert split in ['train', 'val', 'all']
    
    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes,split,cam_ids=camera_names)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    max_episode_len=train_dataset.max_episode_len
    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim,max_episode_len


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

#复制自RISE，用于夹爪宽度量纲统一
def decode_gripper_width(gripper_width):
    return gripper_width / 1000. * 0.095