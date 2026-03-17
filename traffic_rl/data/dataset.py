# =============================================================================
# data/dataset.py  ——  数据加载与预处理
# =============================================================================
# 功能：
#   读取 ConFormer 的 data.npz（shape [T, N, C]），
#   切成 ASTER 风格的三元组窗口：
#       short_term  [T_short, N, C]   — 短期上下文（输入短期编码器）
#       long_term   [T_long,  N, C]   — 长期上下文（输入长期编码器）
#       target      [K_max,   N, 1]   — 二值事件标签（速度异常=1）
# =============================================================================

import os
import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# StandardScaler
# ─────────────────────────────────────────────────────────────────────────────

class StandardScaler:
    """对速度通道做 Z-score 归一化。"""

    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, data):
        """data: 任意形状的 numpy 数组。"""
        self.mean = float(data.mean())
        self.std  = float(data.std())
        return self

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data):
        return data * self.std + self.mean


# ─────────────────────────────────────────────────────────────────────────────
# TrafficDataset
# ─────────────────────────────────────────────────────────────────────────────

class TrafficDataset(Dataset):
    """
    用滑动窗口把 [T, N, C] 的交通数据切成三元组：
        short_term : [T_short, N, C]   输入（速度已归一化）
        long_term  : [T_long,  N, C]   输入（速度已归一化）
        target     : [K_max,   N, 1]   二值事件标签

    事件定义：原始速度 < mean - threshold_sigma * std  → 1，否则 → 0
    """

    def __init__(self, data, T_long, T_short, K_max, scaler, event_threshold):
        """
        Args:
            data            : np.array [T, N, C]，速度通道（0）已经过归一化
            T_long          : 长期窗口长度
            T_short         : 短期窗口长度（≤ T_long）
            K_max           : 未来预测步数
            scaler          : 已 fit 的 StandardScaler（用于把归一化速度还原）
            event_threshold : 原始速度阈值；低于此值视为异常事件
        """
        assert T_short <= T_long, "T_short 必须 ≤ T_long"
        self.data            = data
        self.T_long          = T_long
        self.T_short         = T_short
        self.K_max           = K_max
        self.scaler          = scaler
        self.event_threshold = event_threshold
        self.window          = T_long + K_max   # 每个样本覆盖的总步数

    def __len__(self):
        return max(0, len(self.data) - self.window)

    def __getitem__(self, idx):
        raw = self.data[idx : idx + self.window]   # [window, N, C]

        # ── 输入 ──────────────────────────────────────────────────────────
        long_term  = raw[: self.T_long]                              # [T_long,  N, C]
        short_term = raw[self.T_long - self.T_short : self.T_long]  # [T_short, N, C]

        # ── 目标：未来 K_max 步速度 → 二值标签 ───────────────────────────
        future_norm = raw[self.T_long :, :, 0]                       # [K_max, N] 归一化速度
        future_raw  = self.scaler.inverse_transform(future_norm)     # 还原为原始速度
        target = (future_raw < self.event_threshold).astype(np.float32)  # [K_max, N]
        target = target[:, :, np.newaxis]                            # [K_max, N, 1]

        return (
            torch.FloatTensor(short_term),  # [T_short, N, C]
            torch.FloatTensor(long_term),   # [T_long,  N, C]
            torch.FloatTensor(target),      # [K_max,   N, 1]
        )


# ─────────────────────────────────────────────────────────────────────────────
# build_datasets  —  对外唯一入口
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(cfg):
    """
    读取 data.npz，按 60/20/20 切分，拟合 scaler，
    返回 (train_dataset, val_dataset, test_dataset, scaler, event_threshold)。
    """
    data_path = os.path.join(cfg["data_dir"], "data.npz")
    raw = np.load(data_path)["data"].astype(np.float32)   # [T, N, F]

    T, N, F = raw.shape
    C = min(F, cfg["input_dim"])
    data = raw[..., :C]    # 只取用到的通道

    # 60 / 20 / 20 划分
    train_end = int(0.6 * T)
    val_end   = int(0.8 * T)

    train_raw = data[:train_end]
    val_raw   = data[train_end:val_end]
    test_raw  = data[val_end:]

    # 在训练集速度通道上拟合 scaler
    scaler = StandardScaler().fit(train_raw[..., 0])

    # 事件阈值（基于原始速度，非归一化）
    raw_speed     = train_raw[..., 0]
    event_thr     = raw_speed.mean() - cfg["threshold_sigma"] * raw_speed.std()

    # 归一化速度通道
    def normalize(split):
        s = split.copy()
        s[..., 0] = scaler.transform(s[..., 0])
        return s

    train_data = normalize(train_raw)
    val_data   = normalize(val_raw)
    test_data  = normalize(test_raw)

    T_long  = cfg["T_long"]
    T_short = cfg["T_short"]
    K_max   = cfg["K_max"]

    train_ds = TrafficDataset(train_data, T_long, T_short, K_max, scaler, event_thr)
    val_ds   = TrafficDataset(val_data,   T_long, T_short, K_max, scaler, event_thr)
    test_ds  = TrafficDataset(test_data,  T_long, T_short, K_max, scaler, event_thr)

    print(f"[Data] 节点数={N}  通道数={C}  事件阈值={event_thr:.3f}")
    print(f"[Data] Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")

    return train_ds, val_ds, test_ds, scaler, event_thr
