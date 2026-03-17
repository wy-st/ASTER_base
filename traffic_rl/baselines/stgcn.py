# =============================================================================
# baselines/stgcn.py  ——  STGCN 基线
# =============================================================================
# 参考：Yu et al. (2018) "Spatio-Temporal Graph Convolutional Networks:
#       A Deep Learning Framework for Traffic Forecasting" (ICLR 2018)
#
# 原始 STGCN 需要预计算谱域图（切比雪夫多项式）。这里采用自适应邻接矩阵
# （Graph WaveNet 风格，无需外部图结构），使其在任何数据集上开箱即用。
#
# 结构：
#   输入 [B, T, N, C]
#   → ST-Conv Block 1 (Tconv → AdpGCN → Tconv)
#   → ST-Conv Block 2
#   → 时间维平均池化 → [B, N, model_dim]
#   → EncoderResourcePredictor
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import EncoderResourcePredictor


# ─────────────────────────────────────────────────────────────────────────────
# 时间门控卷积（GLU）
# ─────────────────────────────────────────────────────────────────────────────

class TemporalGatedConv(nn.Module):
    """
    1-D 时间卷积 + GLU 门控。
    输入 [B, in_ch, N, T] → 输出 [B, out_ch, N, T]（same padding）。
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        # same padding in time dimension
        pad = (kernel_size - 1) * dilation // 2
        # Conv2d: 在 T（宽）维做 1-D conv，N（高）维不卷积
        self.conv = nn.Conv2d(
            in_ch, 2 * out_ch,
            kernel_size=(1, kernel_size),
            padding=(0, pad),
            dilation=(1, dilation),
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_ch, N, T]"""
        a, b = self.conv(x).chunk(2, dim=1)   # 各 [B, out_ch, N, T]
        return self.bn(a * torch.sigmoid(b))   # GLU


# ─────────────────────────────────────────────────────────────────────────────
# 自适应图卷积
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveGraphConv(nn.Module):
    """
    基于可学习节点嵌入的自适应图卷积。
    A = softmax(relu(E1 @ E2^T))  in [N, N]
    输入 [B, in_ch, N, T] → 输出 [B, out_ch, N, T]。
    """

    def __init__(self, in_ch: int, out_ch: int, num_nodes: int,
                 emb_dim: int = 10, dropout: float = 0.1):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.W  = nn.Linear(in_ch, out_ch, bias=False)
        self.b  = nn.Parameter(torch.zeros(out_ch))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_ch, N, T]"""
        # 构建自适应邻接
        A = F.softmax(F.relu(self.E1 @ self.E2.T), dim=-1)  # [N, N]

        # [B, in_ch, N, T] → [B, T, N, in_ch]
        x = x.permute(0, 3, 2, 1)
        x = self.drop(self.W(x))                              # [B, T, N, out_ch]
        # 图传播：H' = A H
        x = torch.einsum('nm, btnc -> btmc', A, x) + self.b  # [B, T, N, out_ch]
        return x.permute(0, 3, 2, 1)                          # [B, out_ch, N, T]


# ─────────────────────────────────────────────────────────────────────────────
# ST-Conv Block：Tconv → GCN → Tconv + 残差 + LayerNorm
# ─────────────────────────────────────────────────────────────────────────────

class STConvBlock(nn.Module):
    """
    STGCN 基本单元：Temporal Gated Conv → Adaptive GCN → Temporal Gated Conv。
    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int,
                 num_nodes: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()

        self.tconv1 = TemporalGatedConv(in_ch,   mid_ch, kernel_size)
        self.gcn    = AdaptiveGraphConv(mid_ch,  mid_ch, num_nodes,  dropout=dropout)
        self.tconv2 = TemporalGatedConv(mid_ch,  out_ch, kernel_size)

        # 如果通道数不同，用 1×1 映射残差
        self.residual = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch else nn.Identity()
        )
        self.ln = nn.LayerNorm(out_ch)   # 作用在通道维

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_ch, N, T]"""
        res = self.residual(x)

        out = self.tconv1(x)    # [B, mid_ch, N, T]
        out = F.relu(self.gcn(out))
        out = self.tconv2(out)  # [B, out_ch, N, T]

        out = out + res
        # LayerNorm over channel dim
        out = out.permute(0, 2, 3, 1)   # [B, N, T, out_ch]
        out = self.ln(out)
        out = out.permute(0, 3, 1, 2)   # [B, out_ch, N, T]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# STGCN 编码器
# ─────────────────────────────────────────────────────────────────────────────

class STGCNEncoder(nn.Module):
    """
    STGCN 编码器。

    Input:  [B, T, N, C]
    Output: [B, N, model_dim]
    """

    def __init__(self, T: int, input_dim: int, num_nodes: int,
                 model_dim: int, dropout: float = 0.1):
        super().__init__()
        mid = max(16, model_dim // 2)

        # 两个 ST-Conv Block
        self.block1 = STConvBlock(input_dim, mid,      mid,       num_nodes,
                                  kernel_size=3, dropout=dropout)
        self.block2 = STConvBlock(mid,       model_dim, model_dim, num_nodes,
                                  kernel_size=3, dropout=dropout)

        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, N, C]  →  [B, N, model_dim]"""
        # STGCN 期望 [B, C, N, T]
        x = x.permute(0, 3, 2, 1)        # [B, C, N, T]

        x = self.block1(x)                # [B, mid, N, T]
        x = self.block2(x)                # [B, model_dim, N, T]

        # 时间维平均池化
        x = x.mean(dim=-1)               # [B, model_dim, N]
        x = x.permute(0, 2, 1)           # [B, N, model_dim]
        return self.output_norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# STGCN Predictor
# ─────────────────────────────────────────────────────────────────────────────

class STGCNPredictor(EncoderResourcePredictor):
    """
    STGCN baseline。

    短/长期窗口各用一个独立的 STGCNEncoder，
    后端复用主模型的资源感知融合 + 解码器。
    """

    def __init__(self, cfg: dict):
        D       = cfg["model_dim"]
        N       = cfg["num_nodes"]
        dropout = cfg["dropout"]

        short_enc = STGCNEncoder(
            T=cfg["T_short"], input_dim=cfg["input_dim"],
            num_nodes=N, model_dim=D, dropout=dropout,
        )
        long_enc = STGCNEncoder(
            T=cfg["T_long"], input_dim=cfg["input_dim"],
            num_nodes=N, model_dim=D, dropout=dropout,
        )

        super().__init__(cfg, enc_dim=D,
                         short_encoder=short_enc, long_encoder=long_enc)
