# =============================================================================
# config.py  ——  全局超参数配置
# =============================================================================
# 快速使用：
#   1. 修改 ACTIVE_DATASET  → 切换数据集 ("TKY" / "BA" / "SD")
#   2. 修改 ACTIVE_MODEL    → 切换模型（见下方列表）
#   3. 直接运行: python train.py
#
# ── 可选模型（ACTIVE_MODEL）────────────────────────────────────────────────
#   "our_model"     —— 主模型：ConFormer 风格编码器（无 GCN）+ ASTER RL
#   "conformer_rl"  —— 基线 ：ConFormer 原版编码器（含 GCN + ToD/DoW Emb）+ ASTER RL
#   "aster"         —— 基线 ：ASTER 原版双编码器（Conv1D + MTGNN-lite）+ ASTER RL
#   "fc_lstm"       —— 基线 ：FC-LSTM 编码器 + ASTER RL
#   "stgcn"         —— 基线 ：STGCN 编码器 + ASTER RL
#   "staeformer"    —— 基线 ：STAEformer 编码器 + ASTER RL
#
# ── 数据集说明 ──────────────────────────────────────────────────────────────
#   TKY  —  东京路网，1843 节点，10 分钟间隔
#   BA   —  加州湾区高速，节点数下载后确认，5 分钟间隔
#   SD   —  圣地亚哥高速，节点数下载后确认，5 分钟间隔
#
# ── 关于 input_dim ──────────────────────────────────────────────────────────
#   ConFormer data.npz 均为 3 通道：
#     ch0 = 速度（归一化浮点）
#     ch1 = ToD  (Time-of-Day, 0~1 浮点, ConFormer 用于 Embedding 查表)
#     ch2 = DoW  (Day-of-Week, 0~6 整数, ConFormer 用于 Embedding 查表)
#
#   ConFormer 原始 model_args input_dim=1：仅速度进 input_proj，
#   ch1/ch2 另走 Embedding 查表。
#   我们的编码器（our_model / fc_lstm / stgcn / staeformer / aster）
#   无独立 Embedding 查表，所有通道统一走线性层，因此 input_dim=3。
#   conformer_rl 内部自己区分 ch0/ch1/ch2，外部 input_dim 仍为 3。
#
# ⚠️  BA / SD 的 num_nodes 需下载数据后确认：
#      import numpy as np
#      d = np.load("data/BA/data.npz")["data"]
#      print(d.shape)   # → [T, num_nodes, 3]
# =============================================================================

# ── 切换这两行即可选择实验条件 ──────────────────────────────────────────────
ACTIVE_DATASET = "TKY"
ACTIVE_MODEL   = "our_model"    # 见上方列表

# ── 合法模型名集合（用于 train.py 的断言检查）──────────────────────────────
VALID_MODELS = frozenset(
    ["our_model", "conformer_rl", "aster", "fc_lstm", "stgcn", "staeformer"]
)


CONFIGS = {

    # -------------------------------------------------------------------------
    # TKY  ·  1843 个东京路段
    # data.npz  →  shape [T, 1843, 3]
    #   ch0=速度, ch1=ToD(0~1), ch2=DoW(0~6)
    #   10 分钟间隔 → steps_per_day=144
    # ConFormer yaml：in_steps=6, out_steps=6
    # -------------------------------------------------------------------------
    "TKY": {
        # ── 数据 ──────────────────────────────────────────────────────────
        "data_dir":      "/home/user/ConFormer_-base/data/TKY",
        "num_nodes":     1843,
        "input_dim":     3,          # ch0(速度)+ch1(ToD)+ch2(DoW)
        "steps_per_day": 144,        # 10 分钟间隔

        # ── 时间窗口 / 预测步数 ───────────────────────────────────────────
        "T_long":   24,   # 长期窗口（24 步 = 4 小时）
        "T_short":   6,   # 短期窗口（ 6 步 = 1 小时）
        "K_max":     6,   # 最大预测步数（与 ConFormer out_steps 对齐）

        # ── 事件阈值 ──────────────────────────────────────────────────────
        "threshold_sigma": 0.5,

        # ── 模型结构 ──────────────────────────────────────────────────────
        "model_dim":  64,
        "c_dim":      64,
        "C_common":   32,
        "num_heads":   4,   # 注：TKY 节点多，可降为 2
        "num_layers":  3,
        "dropout":    0.1,

        # ── RL 环境 ───────────────────────────────────────────────────────
        "total_resources": 100,
        "num_actions":       2,      # 0=不调度  1=调度

        # ── DQN 超参 ──────────────────────────────────────────────────────
        "gamma":              0.99,
        "epsilon_start":      1.0,
        "epsilon_decay":      0.995,
        "epsilon_min":        0.05,
        "replay_capacity":    5000,
        "rl_batch_size":      32,
        "target_update_freq": 100,

        # ── 奖励权重 ──────────────────────────────────────────────────────
        "reward_alpha": 1.0,    # 成功率
        "reward_beta":  0.01,   # 误报惩罚
        "reward_gamma": 0.01,   # 距离惩罚
        "reward_delta": 0.3,    # 提前检测奖励

        # ── 训练参数 ──────────────────────────────────────────────────────
        "batch_size":          4,    # 节点多，batch 小
        "num_epochs":          50,
        "lr":                  0.001,
        "early_stop_patience": 10,
        "save_every":          5,    # 每 5 epoch 保存一次检查点

        # ── 其他 ──────────────────────────────────────────────────────────
        "seed":       42,
        "device":     "cuda",
        "results_dir": "results",    # 结果保存根目录
    },

    # -------------------------------------------------------------------------
    # BA  ·  加州湾区高速公路（ConFormer 自建数据集）
    # data.npz  →  shape [T, num_nodes, 3]
    #   5 分钟间隔 → steps_per_day=288
    # ⚠️  下载后确认 num_nodes：
    #      import numpy as np; print(np.load("data/BA/data.npz")["data"].shape)
    # -------------------------------------------------------------------------
    "BA": {
        "data_dir":      "/home/user/ConFormer_-base/data/BA",
        "num_nodes":     None,   # ⚠️ 下载后填入，例如：716
        "input_dim":     3,
        "steps_per_day": 288,

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

        "total_resources": 40,
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
        "save_every":          5,

        "seed":       42,
        "device":     "cuda",
        "results_dir": "results",
    },

    # -------------------------------------------------------------------------
    # SD  ·  圣地亚哥高速公路（ConFormer 自建数据集）
    # data.npz  →  shape [T, num_nodes, 3]
    #   5 分钟间隔 → steps_per_day=288
    # ⚠️  下载后确认 num_nodes：
    #      import numpy as np; print(np.load("data/SD/data.npz")["data"].shape)
    # -------------------------------------------------------------------------
    "SD": {
        "data_dir":      "/home/user/ConFormer_-base/data/SD",
        "num_nodes":     None,   # ⚠️ 下载后填入
        "input_dim":     3,
        "steps_per_day": 288,

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

        "total_resources": 40,
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
        "save_every":          5,

        "seed":       42,
        "device":     "cuda",
        "results_dir": "results",
    },
}

# ── 对外暴露的主配置对象 ─────────────────────────────────────────────────────
CFG = CONFIGS[ACTIVE_DATASET]
