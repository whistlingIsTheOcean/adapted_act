#!/usr/bin/env python
"""
ACT 评估前可视化（不依赖真机 / 不修改任何训练代码）。

用法：
  # 遍历多个【不同】demo，每个 demo 随机取一个时刻，对比预测 vs 真实（RISE inf.py 风格）
  python visualize_predictions.py --ckpt_dir <ckpt目录> --num_samples 10 --save_dir ./vis_out

  # 有显示环境：open3d 交互查看（关掉一个看下一个）
  python visualize_predictions.py --ckpt_dir <ckpt目录> --num_samples 10

  # 固定从某个 demo / 某个时刻开始
  python visualize_predictions.py --ckpt_dir <ckpt目录> --episode_idx 3 --start_ts 50 --num_samples 5

说明：
  - --num_samples = 采样多少个【不同】demo；--start_ts = 指定起点（-1 表示每个 demo 随机取一个时刻）
  - 默认自动选择 ckpt 目录里【最新】的 .ckpt（按修改时间）；也可 --ckpt 指定
  - 无 DISPLAY 且没给 --save_dir 时，自动改用无头渲染并保存到 ./vis_out
  - 图例：
      无头(matplotlib): 绿=真实轨迹、红=预测轨迹，散点颜色=夹爪宽度(蓝闭→红开)
      交互(open3d):     绿=真实轨迹、红=预测轨迹，球大小=夹爪宽度(闭合小/张开大)，含姿态坐标架
  - 数据 tcp 是【全局相机坐标系】，预测/真实同框对比即可
  - 若 load_state_dict 报维度不匹配，说明 --chunk_size/--hidden_dim/--dim_feedforward 与训练不一致，请对齐。
"""

import os
import sys
import json
import pickle
import argparse
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')   # 无头渲染用，不弹窗
import torch
from PIL import Image

REPO = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO, os.path.join(REPO, 'detr')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import open3d as o3d
from constants import SIM_TASK_CONFIGS
from policy import ACTPolicy
from utils.transformation import xyz_rot_transform, rotation_transform
from utils_act import decode_gripper_width

WIDTH_MIN, WIDTH_MAX = 0.0, 0.095   # 米


# ---------- 数据读取（只读，复用训练逻辑） ----------
def list_frame_ids(cam_dir, finish_time):
    return sorted(
        int(os.path.splitext(x)[0])
        for x in os.listdir(cam_dir)
        if os.path.splitext(x)[0].isdigit() and int(os.path.splitext(x)[0]) <= finish_time
    )


def load_episode_actions(demo_dir, camera_names):
    """返回该 demo 的完整动作序列 (N,10)，xyz+rot6d+width，相机坐标系。"""
    meta = json.load(open(os.path.join(demo_dir, 'metadata.json')))
    master_cam = camera_names[0]
    cam_dir = os.path.join(demo_dir, f'cam_{master_cam}')
    frame_ids = list_frame_ids(os.path.join(cam_dir, 'color'), meta['finish_time'])
    tcp_list, width_list = [], []
    for t in frame_ids[:-1]:
        tcp_list.append(np.load(os.path.join(cam_dir, 'tcp', f'{t}.npy'))[:7])
        width_list.append(decode_gripper_width(
            np.load(os.path.join(cam_dir, 'gripper_command', f'{t}.npy'))[0]))
    tcp_arr = xyz_rot_transform(np.stack(tcp_list), from_rep='quaternion', to_rep='rotation_6d')
    action = np.concatenate([tcp_arr, np.stack(width_list)[..., None]], axis=-1)
    return frame_ids, action


def load_images(demo_dir, camera_names, frame_ids, start_ts):
    """返回 (2,3,H,W) float32，0~1。全局相机按时间戳取，腕部相机取最近帧。"""
    master_id = frame_ids[start_ts]
    master = np.array(Image.open(
        os.path.join(demo_dir, f'cam_{camera_names[0]}', 'color', f'{master_id}.png')))
    meta = json.load(open(os.path.join(demo_dir, 'metadata.json')))
    slave_ids = np.array(list_frame_ids(
        os.path.join(demo_dir, f'cam_{camera_names[1]}', 'color'), meta['finish_time']))
    slave_id = slave_ids[np.argmin(np.abs(slave_ids - master_id))]
    slave = np.array(Image.open(
        os.path.join(demo_dir, f'cam_{camera_names[1]}', 'color', f'{slave_id}.png')))
    images = torch.from_numpy(np.stack([master, slave]))   # (2,H,W,3)
    images = torch.einsum('k h w c -> k c h w', images) / 255.0
    return images.float()


# ---------- 模型 ----------
def build_policy(ckpt_path, cfg, device):
    policy_config = {
        'lr': 1e-5, 'num_queries': cfg['chunk_size'], 'kl_weight': cfg['kl_weight'],
        'hidden_dim': cfg['hidden_dim'], 'dim_feedforward': cfg['dim_feedforward'],
        'lr_backbone': 1e-5, 'backbone': 'resnet18', 'enc_layers': cfg['enc_layers'],
        'dec_layers': cfg['dec_layers'], 'nheads': cfg['nheads'],
        'camera_names': cfg['camera_names'], 'state_dim': cfg['state_dim'],
    }
    # ACTPolicy 内部 build_ACT_model_and_optimizer 会 parse sys.argv，
    # 需临时提供 DETR parser 的 required 参数。
    _argv = sys.argv[:]
    sys.argv = ['visualize_predictions', '--ckpt_dir', '.', '--policy_class', 'ACT',
                '--task_name', cfg['task_name'], '--seed', '0', '--num_epochs', '1',
                '--chunk_size', str(cfg['chunk_size']), '--state_dim', str(cfg['state_dim'])]
    try:
        policy = ACTPolicy(policy_config)
    finally:
        sys.argv = _argv
    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing or unexpected:
        print('[warn] missing:', len(missing), ' unexpected:', len(unexpected))
        if missing:
            print('   missing 前 5 个:', missing[:5])
    policy.to(device).eval()
    return policy


# ---------- 可视化 ----------
def rot6d_to_axes(rot6d, length=0.02):
    R = rotation_transform(np.array(rot6d, dtype=np.float32),
                           from_rep='rotation_6d', to_rep='matrix')
    return R


def width_to_color(w):
    """夹爪宽度 -> 颜色：闭合=蓝，张开=红（避免与真实轨迹的绿色混淆）。"""
    t = float(np.clip((w - WIDTH_MIN) / (WIDTH_MAX - WIDTH_MIN), 0, 1))
    return np.array([1 - t, 0.0, t])


def make_traj_geoms(traj, color, every=5, base_radius=0.006, axes_len=0.02,
                    label='', width_scale=True):
    """
    返回 [球, 线, 坐标架] 的几何体列表。

    - 球统一用轨迹色 color（真实=绿 / 预测=红），避免两条轨迹球色混淆；
    - 夹爪宽度用球半径表示（width_scale=True：闭合小球、张开大球）。
    """
    pts = traj[:, :3]
    widths = traj[:, -1]
    spheres = []
    for p, w in zip(pts, widths):
        if width_scale:
            t = float(np.clip((w - WIDTH_MIN) / (WIDTH_MAX - WIDTH_MIN), 0, 1))
            r = base_radius * (0.5 + 1.5 * t)
        else:
            r = base_radius
        s = o3d.geometry.TriangleMesh.create_sphere(r)
        s.translate(p)
        s.paint_uniform_color(color)
        spheres.append(s)
    # 连线
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(pts)
    line.lines = o3d.utility.Vector2iVector(
        [[i, i + 1] for i in range(len(pts) - 1)])
    line.paint_uniform_color(color)
    # 坐标架（每 every 个点）
    axes = []
    for i in range(0, len(traj), every):
        R = rot6d_to_axes(traj[i, 3:9])
        o = pts[i]
        for axis, ac in zip(np.eye(3), [(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
            al = o3d.geometry.LineSet()
            al.points = o3d.utility.Vector3dVector(np.stack([o, o + axes_len * R @ axis]))
            al.lines = o3d.utility.Vector2iVector([[0, 1]])
            al.paint_uniform_color(ac)
            axes.append(al)
    return spheres + [line] + axes


def render_matplotlib(real, pred, path, title):
    """matplotlib 3D 无头渲染（服务器无 DISPLAY 也能用，纯 CPU）。"""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(real[:, 0], real[:, 1], real[:, 2], '-', color='green', lw=2.5, label='real')
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], '-', color='red', lw=2.5, label='pred')
    ax.scatter(*real[0, :3], color='black', s=70, marker='*', label='real start')
    ax.scatter(*pred[0, :3], color='orange', s=70, marker='*', label='pred start')
    sc = ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c=pred[:, -1],
                    cmap='coolwarm', s=14, vmin=WIDTH_MIN, vmax=WIDTH_MAX)
    fig.colorbar(sc, ax=ax, label='gripper width (蓝=闭合, 红=张开)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.legend()
    ax.set_title(title)
    try:
        ax.set_box_aspect((np.ptp(real[:, 0]) + 1e-6,
                           np.ptp(real[:, 1]) + 1e-6,
                           np.ptp(real[:, 2]) + 1e-6))
    except Exception:
        pass
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  已保存: {path}')


def synthesize_ensemble(policy, demo_dir, camera_names, frame_ids, action,
                        start_ts, chunk_size, stats, device, k=0.01):
    """
    模拟 eval_bc 的 temporal_agg：从 start_ts 起每步预测一次 chunk，
    对每个时刻把所有覆盖它的预测做指数加权平均，合成"实际会执行的轨迹"。

    返回 (M,10) 未归一化的动作轨迹，其中第 t 步对应真实轨迹 action[start_ts-1+t]
    （与训练/评估的 start_ts-1 对齐 hack 一致）。
    """
    max_t = min(chunk_size, len(action) - start_ts)
    num_queries = chunk_size
    buffer = np.zeros((max_t, max_t + num_queries, action.shape[1]))
    exec_traj = np.zeros((max_t, action.shape[1]))
    for t in range(max_t):
        qpos = action[start_ts + t]
        images = load_images(demo_dir, camera_names, frame_ids, start_ts + t)
        qpos_norm = (qpos - stats['qpos_mean']) / stats['qpos_std']
        qpos_t = torch.from_numpy(qpos_norm).float().to(device).unsqueeze(0)
        with torch.inference_mode():
            a_hat = policy(qpos_t, images.unsqueeze(0).to(device))[0].cpu().numpy()
        a = a_hat * stats['action_std'] + stats['action_mean']      # (chunk,10)
        buffer[t, t:t + num_queries] = a
        # 聚合第 t 步：所有覆盖 t 的预测按发起时刻越新权重越大
        populated = np.all(buffer[:, t] != 0, axis=1)
        acts = buffer[populated, t]
        weights = np.exp(-k * np.arange(len(acts)))
        weights = weights / weights.sum()
        exec_traj[t] = (acts * weights[:, None]).sum(axis=0)
    return exec_traj


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True, help='训练输出目录')
    parser.add_argument('--ckpt', type=str, default='', help='ckpt 文件名；默认自动选最新的 .ckpt')
    parser.add_argument('--task', type=str, default='real_data')
    parser.add_argument('--episode_idx', type=int, default=0, help='从第几个 demo 开始')
    parser.add_argument('--start_ts', type=int, default=-1, help='起点时间步；-1 = 每个 demo 随机取一点')
    parser.add_argument('--num_samples', type=int, default=5, help='采样多少个【不同】demo')
    parser.add_argument('--temporal_agg', action='store_true',
                        help='合成 temporal ensembling 后的"实际执行轨迹"再对比（模拟真机 --temporal_agg）')
    parser.add_argument('--save_dir', type=str, default='', help='非空则无头渲染 png；留空且有显示则 open3d 交互')
    parser.add_argument('--chunk_size', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=3200)
    parser.add_argument('--kl_weight', type=int, default=10)
    parser.add_argument('--enc_layers', type=int, default=4)
    parser.add_argument('--dec_layers', type=int, default=7)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--state_dim', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 选 ckpt：默认取目录里最新的（按修改时间）
    if args.ckpt:
        ckpt_path = os.path.join(args.ckpt_dir, args.ckpt)
    else:
        ckpts = [os.path.join(args.ckpt_dir, f) for f in os.listdir(args.ckpt_dir)
                 if f.endswith('.ckpt')]
        if not ckpts:
            print(f'[错误] {args.ckpt_dir} 下没有 .ckpt 文件，用 --ckpt 指定或检查 --ckpt_dir')
            sys.exit(1)
        ckpt_path = max(ckpts, key=os.path.getmtime)
    if not os.path.exists(ckpt_path):
        print(f'[错误] 找不到 ckpt: {ckpt_path}\n用 --ckpt 指定，或检查 --ckpt_dir')
        sys.exit(1)
    print(f'加载 ckpt: {ckpt_path}')

    task_cfg = SIM_TASK_CONFIGS[args.task]
    stats_path = os.path.join(args.ckpt_dir, 'dataset_stats.pkl')
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
    else:
        stats = None
    if stats is None:
        # 兜底：临时用 get_norm_stats 现算（较慢）
        from utils_act import get_norm_stats
        print('[warn] 未找到 dataset_stats.pkl，现场统计……')
        stats = get_norm_stats(task_cfg['dataset_dir'], task_cfg['num_episodes'],
                               'train', task_cfg['camera_names'])
    print(f'数据目录: {task_cfg["dataset_dir"]}')

    policy = build_policy(ckpt_path,
                          {**vars(args), 'camera_names': task_cfg['camera_names'],
                           'task_name': args.task},
                          device)
    # 无 DISPLAY 且未指定保存目录 → 自动转无头渲染，避免 open3d GLFW 失败
    if not args.save_dir and not os.environ.get('DISPLAY'):
        args.save_dir = 'vis_out'
        print(f'[info] 未检测到 DISPLAY，自动改用无头渲染，保存到 {args.save_dir}/')

    train_dir = os.path.join(task_cfg['dataset_dir'], 'train')
    demos = sorted(os.listdir(train_dir))

    for s in range(args.num_samples):
        demo_idx = (args.episode_idx + s) % len(demos)
        demo_dir = os.path.join(train_dir, demos[demo_idx])
        frame_ids, action = load_episode_actions(demo_dir, task_cfg['camera_names'])
        ts = min(args.start_ts, len(action) - 1) if args.start_ts >= 0 \
            else int(np.random.randint(0, len(action)))
        print(f'[sample {s}] demo={demos[demo_idx]}  帧数={len(frame_ids)}  start_ts={ts}')

        if args.temporal_agg:
            # 合成 temporal ensembling 后的实际执行轨迹（每步预测 + 指数加权平均）
            pred = synthesize_ensemble(policy, demo_dir, task_cfg['camera_names'],
                                       frame_ids, action, ts, args.chunk_size,
                                       stats, device)
            real = action[max(0, ts - 1):][:len(pred)]
        else:
            # 单次前向：模型在 ts 时刻一次性预测的未来 chunk
            qpos = action[ts]
            images = load_images(demo_dir, task_cfg['camera_names'], frame_ids, ts)
            qpos_norm = (qpos - stats['qpos_mean']) / stats['qpos_std']
            qpos_t = torch.from_numpy(qpos_norm).float().to(device).unsqueeze(0)
            with torch.inference_mode():
                a_hat = policy(qpos_t, images.unsqueeze(0).to(device))[0].cpu().numpy()
            pred = a_hat * stats['action_std'] + stats['action_mean']   # (chunk, 10)
            # 真实轨迹（与训练一致的 start_ts-1 hack）
            real = action[max(0, ts - 1):]

        # 关键点重合度指标（只看真实长度内的对齐）
        L = min(len(real), len(pred))
        pos_err = np.linalg.norm(pred[:L, :3] - real[:L, :3], axis=-1)
        print(f'\n[sample {s}] start_ts={ts}  pred_len={len(pred)}  real_len={len(real)}')
        print(f'  位置误差(相机坐标,m): 均值={pos_err.mean():.4f}  最大={pos_err.max():.4f}')

        title = f'demo={os.path.basename(demo_dir)} ts={ts}' + \
            ('  [temporal_agg]' if args.temporal_agg else '')
        geoms = make_traj_geoms(real, color=(0.2, 0.8, 0.2), label='real')
        geoms += make_traj_geoms(pred, color=(0.9, 0.2, 0.2), label='pred')
        # 起点标记
        for c, o in [((0, 0, 0), real[0, :3]), ((0.5, 0.5, 0.5), pred[0, :3])]:
            sp = o3d.geometry.TriangleMesh.create_sphere(0.012)
            sp.translate(o)
            sp.paint_uniform_color(c)
            geoms.append(sp)

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            render_matplotlib(real, pred,
                              os.path.join(args.save_dir, f'sample{s}_ts{ts}.png'),
                              title)
        else:
            o3d.visualization.draw_geometries(
                geoms, window_name=title, width=1280, height=720)
            print('  关闭窗口查看下一个……')


if __name__ == '__main__':
    main()
