# =============================================================================
# config.py  ——  全局超参数配置
# =============================================================================
# 使用方法：
#   1. 把 ACTIVE_DATASET 改成你要跑的数据集 ("TKY" / "BA" / "SD")
#   2. 确认 data_dir 路径指向正确的 data.npz 所在目录
#   3. 直接运行: python train.py
#
# ConFormer 三个数据集：
#   TKY  —  东京路网        data/TKY/data.npz   1843节点  10分钟间隔
#   BA   —  加州湾区高速    data/BA/data.npz    下载后确认
#   SD   —  圣地亚哥高速    data/SD/data.npz    下载后确认
#
# ── 关于 input_dim 的说明 ──────────────────────────────────────────────────
# ConFormer 的 data.npz 有 3 个通道：
#   ch 0 : 速度（归一化后喂入线性投影）
#   ch 1 : 时刻 ToD（Time-of-Day，0~1 浮点，用于 tod_embedding 查表）
#   ch 2 : 星期 DoW（Day-of-Week，0~6 整数，用于 dow_embedding 查表）
#
# ConFormer 原始 model_args 里 input_dim=1，意思是 input_proj 只用 ch0，
# ch1/ch2 另走 Embedding 查表。
#
# 我们的 TrafficEncoder 没有 Embedding 查表，所有通道都走线性层，
# 因此 input_dim=3（把速度 + ToD + DoW 一起作为原始特征输入）。
#
# ⚠️  BA / SD 的 num_nodes 需下载数据后确认：
#      import numpy as np
#      d = np.load("data.npz")["data"]
#      print(d.shape)   # → [T, num_nodes, 3]
# =============================================================================

# ── 切换数据集只改这一行 ────────────────────────────────────────────────────
ACTIVE_DATASET = "TKY"


CONFIGS = {

    # -------------------------------------------------------------------------
    # TKY  ·  1843 个东京路段
    # data.npz  →  shape [T, 1843, 3]
    #   ch0=速度, ch1=ToD(0~1), ch2=DoW(0~6)
    #   10 分钟间隔 → steps_per_day=144
    # ConFormer yaml 参数：in_steps=6, out_steps=6, input_dim=1(仅速度进投影)
    #   tod_embedding_dim=32, dow_embedding_dim=32（ch1/ch2 走 Embedding 查表）
    # -------------------------------------------------------------------------
    "TKY": {
        "data_dir":      "/home/user/ConFormer_-base/data/TKY",
        "num_nodes":     1843,
        "input_dim":     3,          # ch0(速度) + ch1(ToD) + ch2(DoW)，全部走线性层
        "steps_per_day": 144,        # 10 分钟间隔 → 每天 144 步

        "T_long":   24,   # 长期窗口（24步 = 4小时）
        "T_short":   6,   # 短期窗口（6步 = 1小时）
        "K_max":     6,   # 最大预测步数（与 ConFormer out_steps=6 对齐）

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
    # BA  ·  加州湾区高速公路（ConFormer 自建数据集）
    # data.npz  →  shape [T, num_nodes, 3]
    #   ch0=速度, ch1=ToD(0~1), ch2=DoW(0~6)
    #   5 分钟间隔 → steps_per_day=288（与加州 PEMS 数据集一致）
    # ⚠️  下载数据后执行：
    #      import numpy as np; d=np.load("data/BA/data.npz")["data"]; print(d.shape)
    #   确认 num_nodes，然后填入下方
    # -------------------------------------------------------------------------
    "BA": {
        "data_dir":      "/home/user/ConFormer_-base/data/BA",
        "num_nodes":     None,  # ⚠️ 下载后填入，例如：716
        "input_dim":     3,     # ch0(速度) + ch1(ToD) + ch2(DoW)
        "steps_per_day": 288,   # 5 分钟间隔

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
    # SD  ·  圣地亚哥高速公路（ConFormer 自建数据集）
    # data.npz  →  shape [T, num_nodes, 3]
    #   ch0=速度, ch1=ToD(0~1), ch2=DoW(0~6)
    #   5 分钟间隔 → steps_per_day=288
    # ⚠️  下载数据后执行：
    #      import numpy as np; d=np.load("data/SD/data.npz")["data"]; print(d.shape)
    #   确认 num_nodes，然后填入下方
    # -------------------------------------------------------------------------
    "SD": {
        "data_dir":      "/home/user/ConFormer_-base/data/SD",
        "num_nodes":     None,  # ⚠️ 下载后填入
        "input_dim":     3,     # ch0(速度) + ch1(ToD) + ch2(DoW)
        "steps_per_day": 288,   # 5 分钟间隔

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
