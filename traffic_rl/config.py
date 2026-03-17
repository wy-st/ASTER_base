# =============================================================================
# config.py  ——  全局超参数配置
# =============================================================================
# 使用方法：
#   1. 把 ACTIVE_DATASET 改成你要跑的数据集 ("TKY" / "BA" / "SD")
#   2. 确认 data_dir 路径指向正确的 data.npz 所在目录
#   3. 直接运行: python train.py
#
# ConFormer 三个数据集：
#   TKY  —  东京路网        data/TKY/data.npz
#   BA   —  加州湾区高速    data/BA/data.npz
#   SD   —  圣地亚哥高速    data/SD/data.npz
#
# ⚠️  BA / SD 的 num_nodes 需下载数据后确认：
#      import numpy as np
#      d = np.load("data.npz")["data"]
#      print(d.shape)   # → [T, num_nodes, C]
# =============================================================================

# ── 切换数据集只改这一行 ────────────────────────────────────────────────────
ACTIVE_DATASET = "TKY"


CONFIGS = {

    # -------------------------------------------------------------------------
    # TKY  ·  1843 个东京路段
    # data.npz  →  shape [T, 1843, 1]   (仅速度，10分钟间隔)
    # -------------------------------------------------------------------------
    "TKY": {
        "data_dir":      "/home/user/ConFormer_-base/data/TKY",
        "num_nodes":     1843,
        "input_dim":     1,          # 只有速度通道
        "steps_per_day": 144,        # 10 分钟间隔 → 每天 144 步

        "T_long":   24,   # 长期窗口（24步 = 4小时）
        "T_short":   6,   # 短期窗口（6步 = 1小时）
        "K_max":     6,   # 最大预测步数

        "threshold_sigma": 0.5,

        "model_dim":  64,
        "c_dim":      64,
        "C_common":   32,
        "num_heads":   2,   # 节点多，减少头数
        "num_layers":  3,
        "dropout":    0.1,

        "total_resources": 100,
        "num_actions":       2,

        "gamma":              0.99,
        "epsilon_start":      1.0,
        "epsilon_decay":      0.995,
        "epsilon_min":        0.05,
        "replay_capacity":    5000,
        "rl_batch_size":      32,
        "target_update_freq": 100,

        "reward_alpha": 1.0,
        "reward_beta":  0.01,
        "reward_gamma": 0.01,
        "reward_delta": 0.3,

        "batch_size":          4,    # 节点多，batch 小一点
        "num_epochs":          50,
        "lr":                  0.001,
        "early_stop_patience": 10,

        "seed":   42,
        "device": "cuda",
    },

    # -------------------------------------------------------------------------
    # BA  ·  加州湾区高速公路
    # data.npz  →  shape [T, num_nodes, C]
    # ⚠️  下载数据后用 np.load("data.npz")["data"].shape 确认 num_nodes
    # -------------------------------------------------------------------------
    "BA": {
        "data_dir":      "/home/user/ConFormer_-base/data/BA",
        "num_nodes":     None,  # ⚠️ 下载数据后填入，例如：325
        "input_dim":     3,     # 下载后确认（速度 + ToD + DoW，或仅速度）
        "steps_per_day": 288,   # 5分钟间隔（下载后确认）

        "T_long":   36,
        "T_short":  12,
        "K_max":    12,

        "threshold_sigma": 0.5,

        "model_dim":  64,
        "c_dim":      64,
        "C_common":   32,
        "num_heads":   4,
        "num_layers":  3,
        "dropout":    0.1,

        "total_resources": 40,  # 根据节点数酌情调整
        "num_actions":      2,

        "gamma":              0.99,
        "epsilon_start":      1.0,
        "epsilon_decay":      0.995,
        "epsilon_min":        0.05,
        "replay_capacity":    5000,
        "rl_batch_size":      32,
        "target_update_freq": 100,

        "reward_alpha": 1.0,
        "reward_beta":  0.01,
        "reward_gamma": 0.01,
        "reward_delta": 0.3,

        "batch_size":          16,
        "num_epochs":          50,
        "lr":                  0.001,
        "early_stop_patience": 10,

        "seed":   42,
        "device": "cuda",
    },

    # -------------------------------------------------------------------------
    # SD  ·  圣地亚哥高速公路
    # data.npz  →  shape [T, num_nodes, C]
    # ⚠️  下载数据后用 np.load("data.npz")["data"].shape 确认 num_nodes
    # -------------------------------------------------------------------------
    "SD": {
        "data_dir":      "/home/user/ConFormer_-base/data/SD",
        "num_nodes":     None,  # ⚠️ 下载数据后填入
        "input_dim":     3,     # 下载后确认
        "steps_per_day": 288,   # 下载后确认

        "T_long":   36,
        "T_short":  12,
        "K_max":    12,

        "threshold_sigma": 0.5,

        "model_dim":  64,
        "c_dim":      64,
        "C_common":   32,
        "num_heads":   4,
        "num_layers":  3,
        "dropout":    0.1,

        "total_resources": 40,  # 根据节点数酌情调整
        "num_actions":      2,

        "gamma":              0.99,
        "epsilon_start":      1.0,
        "epsilon_decay":      0.995,
        "epsilon_min":        0.05,
        "replay_capacity":    5000,
        "rl_batch_size":      32,
        "target_update_freq": 100,

        "reward_alpha": 1.0,
        "reward_beta":  0.01,
        "reward_gamma": 0.01,
        "reward_delta": 0.3,

        "batch_size":          16,
        "num_epochs":          50,
        "lr":                  0.001,
        "early_stop_patience": 10,

        "seed":   42,
        "device": "cuda",
    },
}

# train.py 直接 from config import CFG 即可
CFG = CONFIGS[ACTIVE_DATASET]
