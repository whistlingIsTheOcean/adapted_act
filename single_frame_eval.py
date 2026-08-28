import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils_act import load_data # data functions
from utils_act import sample_box_pose, sample_insertion_pose # robot functions
from utils_act import compute_dict_mean, set_seed, detach_dict,decode_gripper_width # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos


from PIL import Image
#控制我们的机器
# from eval_agent import Agent,Agent_slave
from utils.constants import *
from dataset.projector import Projector
from utils.ensemble import EnsembleBuffer
from utils.transformation import rotation_transform ,rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform

import IPython
e = IPython.embed

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    
    # get task parameters
    #复用sim通路的储存位置和逻辑，储存我们的数据集的参数
    #is_sim = task_name[:4] == 'sim_'
    
    from constants import SIM_TASK_CONFIGS
    task_config = SIM_TASK_CONFIGS[task_name]
    args['calib']=task_config['calib']
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = args['state_dim']
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         
                         'state_dim':args['state_dim'],
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': True,
        
        'calib':args['calib'],
        #'ensemble_mode':args['ensemble_mode'],
        'discretize_rotation':args['discretize_rotation'],
        'visual':args['visual']
    }

    
    
    if is_eval:
        # 读取训练时保存的 max_episode_len
        #max_len_path = os.path.join(ckpt_dir, 'max_episode_len.txt')
        max_episode_len =158 #代码需要改变存放位置，所以硬编码罢
        config['episode_len'] = max_episode_len

        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            avg_return = eval_bc(config, ckpt_name, save_episode=True)
            #results.append([ckpt_name,  avg_return])

        # for ckpt_name,  avg_return in results:
        #     print(f'{ckpt_name}:  {avg_return=}')
        print()
        exit()
    else:
        print("This script is only for single frame evaluation")
        exit()


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(obs_dict,camera_names):
    curr_images = [] 
    for cam_name in camera_names:
        curr_image = rearrange(obs_dict[cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']-1+20#多20步冗余用来容错
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    post_process = lambda a: a * stats['action_std'] + stats['action_mean']


    

    query_frequency = int(policy_config['num_queries']/5)
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    #真机只做一次实验
    prev_width = None
    num_rollouts = 1
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        

        ### no onscreen render
        

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()
        
        rewards = []
        obs_dict={}
        #qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        #rewards = []
        pred_traj = []   # --visual 时收集预测轨迹（无 visual 时空列表，不 append 不画图）
        with torch.inference_mode():
            for i in range(1,11):
                scene_path=os.path.join(f"/home/yuxuan/task_0003_cue_mismatch_rise/task_0003_user_0001_scene_000{i}_cfg_0001")
                if i==10:
                    scene_path=os.path.join(f"/home/yuxuan/task_0003_cue_mismatch_rise/task_0003_user_0001_scene_0010_cfg_0001")
                fold_list=["approach_pregrasp","grasp_place","grasp_transport"]
                    #for t in range(max_timesteps):
                for fold_name in fold_list:    
                    master_image_path=os.path.join(scene_path,fold_name,"observation_color.png")
                    slave_image_path=os.path.join(scene_path,fold_name,"inhand_observation_color.png")
                    obs_dict['images']={}
                    master_image=np.array(Image.open(master_image_path))
                    slave_image=np.array(Image.open(slave_image_path))
                    obs_dict['images'][camera_names[0]]=master_image
                    obs_dict['images'][camera_names[1]]=slave_image
                    image_list.append(obs_dict['images'])
                    
                    curr_image = get_image(obs_dict['images'], camera_names)


                    ### query policy
                    if config['policy_class'] == "ACT":
                       
                        all_actions = policy(curr_image)  # shape: (num_queries=chunk_size, batch=1, state_dim)
                        assert isinstance(all_actions, torch.Tensor), "返回 action tensor"
                        
                    elif config['policy_class'] == "CNNMLP":
                        raise NotImplementedError
                       
                    else:
                        raise NotImplementedError

                    ### post-process actions
                    raw_action = all_actions.squeeze(1).cpu().numpy()  # (chunk_size, state_dim)
                    action = post_process(raw_action)
                    # 训练数据的 tcp 是 base 坐标系，预测 action 即为 base 坐标
                    #但
                    with open(os.path.join(scene_path,fold_name,f"actions_predict.npy"),'wb')  as f:
                            np.save(f,action)
                    

                    print(f'task={i},shape of action: {action.shape}')
                    

                    if config.get('visual', False):
                        pred_traj.append(action.copy())

                    

                #plt.close()
        

        #rewards = np.array(rewards)
        #episode_return = np.sum(rewards[rewards!=None])
        #episode_returns.append(episode_return)
        
        # if save_episode:
        #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

        # --visual：评估结束后把收集的预测轨迹画成 3D 图（不发控制，仅可视化）
        if config.get('visual', False) and pred_traj:
            pred_traj_arr = np.array(pred_traj)   # (N, chunk_size, 10): xyz + rot6d + width
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            # 每个观测对应一条 chunk_size 长度的未来预测轨迹
            for k, chunk in enumerate(pred_traj_arr):
                ax.plot(chunk[:, 0], chunk[:, 1], chunk[:, 2], lw=1.5, alpha=0.8)
            # 用最后一条 chunk 的散点 + colorbar 表达夹爪宽度
            last = pred_traj_arr[-1]
            sc = ax.scatter(last[:, 0], last[:, 1], last[:, 2],
                            c=last[:, -1], cmap='coolwarm', vmin=0, vmax=0.095)
            fig.colorbar(sc, ax=ax, label='gripper width (m)')
            ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
            ax.set_title(f'Predicted traj (visual) - {ckpt_name}')
            out_png = os.path.join(ckpt_dir, f'pred_traj_visual{rollout_id}.png')
            fig.savefig(out_png, dpi=110, bbox_inches='tight')
            plt.close(fig)
            print(f'[visual] 预测轨迹已保存: {out_png}  ({len(pred_traj)} chunks)')


    #avg_return = np.mean(episode_returns)
    
    #return  avg_return


def forward_pass(data, policy):
    image_data, action_data, is_pad = data
    image_data, action_data, is_pad = image_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)


        if epoch % 50 == 0:# TODO 记得改回100
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def rot_diff(rot1, rot2):
    rot1_mat = rotation_transform(
        rot1,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    rot2_mat = rotation_transform(
        rot2,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    diff = rot1_mat @ rot2_mat.T
    diff = np.diag(diff).sum()
    diff = min(max((diff - 1) / 2.0, -1), 1)
    return np.arccos(diff)

def discretize_rotation(rot_begin, rot_end, rot_step_size = np.pi / 16):
    n_step = int(rot_diff(rot_begin, rot_end) // rot_step_size) + 1
    rot_steps = []
    for i in range(n_step):
        rot_i = rot_begin * (n_step - 1 - i) / n_step + rot_end * (i + 1) / n_step
        rot_steps.append(rot_i)
    return rot_steps



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--visual', action='store_true', help='真机评估时可视化预测轨迹（只采集预测并画图，不发控制指令）')
    
    #迁移到真实机器新增参数。
    # TODO 记得传入calib路径 并决定是否使用离散化旋转
    
    parser.add_argument('--state_dim',default=10,type=int,help="训练时，机器人的状态空间维度")
    parser.add_argument('--calib', action = 'store', type = str, help = 'calibration path', required = False)
    # parser.add_argument('--ensemble_mode', action = 'store', type = str, help = 'temporal ensemble mode', required = False, default = 'new')
    parser.add_argument('--discretize_rotation', action = 'store_true', help = 'whether to discretize rotation process.')
    
    main(vars(parser.parse_args()))
