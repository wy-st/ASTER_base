# =============================================================================
# config.py  ——  全局超参数配置
# =============================================================================
# 使用方法：
#   1. 把 ACTIVE_DATASET 改成你要跑的数据集 ("METRLA" / "PEMSBAY" / "TKY")
#   2. 把 data_dir 改成你的 data.npz 所在路径
#   3. 直接运行: python train.py
# =============================================================================

# ── 切换数据集只改这一行 ────────────────────────────────────────────────────
ACTIVE_DATASET = "METRLA"


CONFIGS = {

    # -------------------------------------------------------------------------
    # METR-LA  ·  207 个洛杉矶路网传感器
    # data.npz  →  shape [T, 207, 3]   (速度 / 时刻ToD / 星期DoW)
    # -------------------------------------------------------------------------
    "METRLA": {
        # 数据
        "data_dir":      "/home/user/ConFormer_-base/data/METRLA",
        "num_nodes":     207,
        "input_dim":     3,          # 通道数：速度 + ToD + DoW
        "steps_per_day": 288,        # 5 分钟间隔 → 每天 288 步

        # 滑动窗口
        "T_long":   36,   # 长期上下文窗口（36步 = 3小时）
        "T_short":  12,   # 短期上下文窗口（12步 = 1小时）
        "K_max":    12,   # 最大预测/调度步数（1小时）

        # 事件检测阈值（速度低于 mean - sigma*std 视为异常事件）
        "threshold_sigma": 0.5,

        # 模型结构
        "model_dim":  64,   # 编码器隐藏维度
        "c_dim":      64,   # GLN 条件向量维度
        "C_common":   32,   # 融合后的公共隐藏维度
        "num_heads":   4,
        "num_layers":  3,
        "dropout":    0.1,

        # 资源调度
        "total_resources": 30,   # 总资源量（如急救车/维修队数量）
        "num_actions":      2,   # 每节点动作空间：{0=不动, 1=调度}

        # DQN / 强化学习
        "gamma":              0.99,
        "epsilon_start":      1.0,
        "epsilon_decay":      0.995,
        "epsilon_min":        0.05,
        "replay_capacity":    5000,
        "rl_batch_size":      32,
        "target_update_freq": 100,

        # 奖励权重
        "reward_alpha": 1.0,    # 成功调度权重
        "reward_beta":  0.01,   # 误报惩罚
        "reward_gamma": 0.01,   # 调度距离惩罚
        "reward_delta": 0.3,    # 早期发现奖励

        # 训练
        "batch_size":          16,
        "num_epochs":          50,
        "lr":                  0.001,
        "early_stop_patience": 10,

        # 其他
        "seed":   42,
        "device": "cuda",   # "cuda" 或 "cpu"
    },

    # -------------------------------------------------------------------------
    # PEMS-BAY  ·  325 个旧金山湾区传感器
    # data.npz  →  shape [T, 325, 3]
    # -------------------------------------------------------------------------
    "PEMSBAY": {
        "data_dir":      "/home/user/ConFormer_-base/data/PEMSBAY",
        "num_nodes":     325,
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

        "seed":   42,
        "device": "cuda",
    },

    # -------------------------------------------------------------------------
    # TKY  ·  1843 个东京路段
    # data.npz  →  shape [T, 1843, 1]   (仅速度)
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
}

# train.py 直接 from config import CFG 即可
CFG = CONFIGS[ACTIVE_DATASET]
