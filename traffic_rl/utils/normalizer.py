# =============================================================================
# utils/normalizer.py  ——  奖励向量归一化（来自 ASTER）
# =============================================================================
# 用滑动均值和方差对多维奖励向量做 Z-score 归一化，
# 防止 Q 网络的学习目标在不同维度上量纲差异过大。
# =============================================================================

import torch


class RewardNormalizer:
    """
    对 D 维奖励向量做在线（online）均值/方差归一化。

    使用指数移动平均（EMA）持续更新统计量，适合非平稳奖励分布。
    最后对归一化后的向量再做 L2 归一化，压制极端值。
    """

    def __init__(self, dim=4, momentum=0.01, eps=1e-6, device="cpu"):
        """
        dim      : 奖励向量维度（对应 4 个目标）
        momentum : EMA 更新系数（越小，历史越重要）
        eps      : 数值稳定项
        device   : "cpu" 或 "cuda"
        """
        self.mean     = torch.zeros(dim, device=device)
        self.var      = torch.ones(dim,  device=device)
        self.momentum = momentum
        self.eps      = eps

    def update(self, reward_vec):
        """用新的奖励向量更新均值和方差。reward_vec: [D]"""
        self.mean = (1 - self.momentum) * self.mean + self.momentum * reward_vec
        self.var  = (1 - self.momentum) * self.var  + self.momentum * (reward_vec - self.mean) ** 2

    def normalize(self, reward_vec):
        """Z-score 归一化后做 L2 正则化，返回归一化后的向量 [D]。"""
        z = (reward_vec - self.mean) / (self.var.sqrt() + self.eps)
        return z / (z.norm() + self.eps)
