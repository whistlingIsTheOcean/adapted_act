#!/usr/bin/env python
"""
训练流程冒烟测试（adapted_act）。

对训练主链路做分层检查，方便定位问题在哪一层：
  1) data   数据加载：load_data -> 取一个 batch，核对各张量形状 / 归一化统计量
  2) model  ACTPolicy 训练模式 forward + backward，核对 loss 有限、梯度存在
  3) e2e    端到端：用真实数据跑 train_bc 一个 epoch

用法：
  python test_train.py            # 全部
  python test_train.py data       # 只测数据
  python test_train.py model      # 只测模型
  python test_train.py e2e        # 只测端到端 1 epoch

说明：
  - 若环境中缺 h5py / IPython，脚本会自动注入 stub，让测试可以跑到真正关心的逻辑。
  - 需要 GPU（模型会 .cuda()）。
"""

import os
import sys
import json
import types
import tempfile

import numpy as np

# 保证能从仓库根目录 import 到包。
# detr/ 也加入 sys.path：detr/models/backbone.py 里有 `from util.misc import ...`
# 的绝对导入，原版 ACT 靠 `pip install -e ./detr` 把 detr 目录放进 sys.path，
# 这里直接加路径等效替代。
REPO = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO, os.path.join(REPO, 'detr')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TASK = 'real_data'          # constants.py 里你要用的任务项
BATCH = 2


# ---------- 1) 缺依赖时注入 stub，让测试能跑 ----------
def _stub_missing_deps():
    for name, stub in [
        ('h5py', types.SimpleNamespace(File='', attrs={})),
        ('IPython', types.SimpleNamespace(embed=lambda *a, **k: None)),
    ]:
        try:
            __import__(name)
        except ImportError:
            sys.modules.setdefault(name, stub)
            print(f"[info] 已注入 stub: {name}（环境中缺失）")


# ---------- 2) 数据加载测试 ----------
def test_data():
    from constants import SIM_TASK_CONFIGS
    from utils_act import load_data

    cfg = SIM_TASK_CONFIGS[TASK]
    print(f"\n==== [data] dataset_dir = {cfg['dataset_dir']} ====")

    train_dl, val_dl, stats, is_sim, max_len = load_data(
        cfg['dataset_dir'], cfg['num_episodes'], cfg['camera_names'],
        BATCH, BATCH)

    # 归一化统计量形状
    for k in ['action_mean', 'action_std']:
        v = np.asarray(stats[k])
        print(f"  stats['{k}'] shape={v.shape}")
        assert v.shape == (10,), f"stats['{k}'] 应为 (10,)，实际 {v.shape}"
    assert max_len >= 2, f"max_episode_len 异常: {max_len}"
    print(f"  max_episode_len = {max_len}")

    # 取一个 batch，核对形状
    image, action, is_pad = next(iter(train_dl))
    print(f"  image  {tuple(image.shape)}  dtype={image.dtype}")
    print(f"  action {tuple(action.shape)}  dtype={action.dtype}")
    print(f"  is_pad {tuple(is_pad.shape)}  dtype={is_pad.dtype}")

    bs, ncam, ch, h, w = image.shape
    assert ncam == len(cfg['camera_names']) == 2, "相机数应为 2"
    assert ch == 3, "图像通道应为 3"
    assert image.min() >= 0 and image.max() <= 1, "图像应已 /255 归一化到 [0,1]"
    assert action.shape[1] == max_len - 1 and action.shape[2] == 10
    assert is_pad.shape == (bs, max_len - 1)
    assert torch_isfinite(action), "action 含 NaN/Inf"
    print("[data] PASS\n")


def torch_isfinite(t):
    import torch
    return bool(torch.isfinite(t).all())


# ---------- 3) 模型 forward/backward 测试 ----------
def test_model():
    import torch
    from policy import ACTPolicy

    print("\n==== [model] ACTPolicy forward + backward ====")
    # build_ACT_model_and_optimizer 内部会 parse sys.argv，需提供 required 参数
    sys.argv = ['test', '--ckpt_dir', tempfile.mkdtemp(prefix='test_ckpt_'),
                '--policy_class', 'ACT', '--task_name', TASK,
                '--seed', '0', '--num_epochs', '1']

    policy_config = {
        'lr': 1e-5, 'num_queries': 8, 'kl_weight': 1, 'hidden_dim': 128,
        'dim_feedforward': 256, 'lr_backbone': 1e-5, 'backbone': 'resnet18',
        'enc_layers': 1, 'dec_layers': 1, 'nheads': 4,
        'camera_names': ['104122063550', '043322070878'],
        'state_dim': 10,
    }
    policy = ACTPolicy(policy_config).cuda()
    policy.train()

    bs, chunk, h, w = BATCH, 8, 128, 128
    image = torch.rand(bs, 2, 3, h, w, device='cuda')          # 0~1 图像
    action = torch.randn(bs, chunk, 10, device='cuda')
    is_pad = torch.zeros(bs, chunk, dtype=torch.bool, device='cuda')

    loss_dict = policy(image, action, is_pad)
    loss = loss_dict['loss']
    print(f"  loss={loss.item():.4f}  l1={loss_dict['l1'].item():.4f}  "
          f"kl={loss_dict['kl'].item():.4f}")
    assert torch.isfinite(loss), "loss 不是有限值"

    loss.backward()
    n_grad = sum(1 for p in policy.parameters() if p.grad is not None)
    print(f"  有梯度的参数数: {n_grad}/{sum(1 for _ in policy.parameters())}")
    assert n_grad > 0, "backward 后没有梯度"
    print("[model] PASS\n")


# ---------- 4) 端到端 1 epoch 训练 ----------
def test_e2e():
    import torch
    from constants import SIM_TASK_CONFIGS
    from utils_act import load_data
    from imitate_episodes import train_bc

    print("\n==== [e2e] train_bc 跑 1 个 epoch ====")
    cfg = SIM_TASK_CONFIGS[TASK]
    train_dl, val_dl, stats, _, max_len = load_data(
        cfg['dataset_dir'], cfg['num_episodes'], cfg['camera_names'], BATCH, BATCH)

    sys.argv = ['test', '--ckpt_dir', tempfile.mkdtemp(prefix='test_ckpt_'),
                '--policy_class', 'ACT', '--task_name', TASK,
                '--seed', '0', '--num_epochs', '1']

    policy_config = {
        'lr': 1e-5, 'num_queries': 8, 'kl_weight': 1, 'hidden_dim': 128,
        'dim_feedforward': 256, 'lr_backbone': 1e-5, 'backbone': 'resnet18',
        'enc_layers': 1, 'dec_layers': 1, 'nheads': 4,
        'camera_names': cfg['camera_names'], 'state_dim': 10,
    }
    config = {
        'num_epochs': 1, 'ckpt_dir': tempfile.mkdtemp(prefix='test_ckpt_'),
        'episode_len': max_len, 'state_dim': 10, 'lr': 1e-5,
        'policy_class': 'ACT', 'onscreen_render': False,
        'policy_config': policy_config, 'task_name': TASK, 'seed': 0,
        'temporal_agg': False, 'camera_names': cfg['camera_names'],
    }
    best_epoch, min_val_loss, best_state = train_bc(train_dl, val_dl, config)
    print(f"  best_epoch={best_epoch}  min_val_loss={min_val_loss:.4f}")
    assert torch.isfinite(torch.tensor(min_val_loss)), "验证损失异常"
    print("[e2e] PASS\n")


if __name__ == '__main__':
    _stub_missing_deps()
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'data'):
        test_data()
    if which in ('all', 'model'):
        test_model()
    if which in ('all', 'e2e'):
        test_e2e()
    print("全部通过 ✔")
