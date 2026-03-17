# =============================================================================
# baselines/aster.py  ——  ASTER 原始模型（简化版）
# =============================================================================
# 参考：ASTER 论文 (wy-st/ASTER_base)
#
# ASTER 的编码器由两部分组成：
#   ShortTermEncoder  : 1-D 时间卷积 + 批归一化 + 空间卷积
#   LongTermEncoder   : 多层扩张因果卷积 + 自适应图卷积（简化 MTGNN）
#
# 与原始 ASTER 的差异：
#   原版 LongTermEncoder 采用 MTGNN（含 graph_constructor / mixprop /
#   dilated_inception 等复杂模块），这里用更轻量的实现替代，
#   功能等价（自适应邻接 + 扩张时间卷积），且无需安装额外依赖。
#
# 接口与主模型一致：返回 (predictions, fused, k)。
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import EncoderResourcePredictor


# ─────────────────────────────────────────────────────────────────────────────
# ShortTermEncoder  —  直接移植自 ASTER models/layer.py
# ─────────────────────────────────────────────────────────────────────────────

class ASTERShortEncoder(nn.Module):
    """
    ASTER 短期编码器（1-D 时间卷积 + 空间卷积）。

    原始 ASTER 论文 ShortTermEncoder 实现。

    Input:  [B, T, N, C]
    Output: [B, N, d_short]
    """

    def __init__(self, T: int, input_dim: int, num_nodes: int,
                 d_short: int = 32, kernel_size: int = 3, num_layers: int = 2):
        super().__init__()
        # ASTER 原版只用 ch0（速度），这里沿用相同设定
        self.input_proj = nn.Conv2d(1, d_short, kernel_size=1)

        # 时间方向残差卷积堆
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_short, d_short,
                          kernel_size=(1, kernel_size),
                          padding=(0, kernel_size // 2)),
                nn.BatchNorm2d(d_short),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_layers)
        ])

        # 空间方向 1-D 卷积（节点间）
        self.spatial_conv = nn.Conv2d(
            d_short, d_short,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
        )

        self.norm = nn.LayerNorm(d_short)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, N, C]  →  [B, N, d_short]"""
        B, T, N, C = x.shape

        # ASTER 只用 ch0：[B, T, N, 1] → [B, 1, T, N]（Conv2d 格式 BCHW）
        x = x[..., 0:1].permute(0, 3, 1, 2)   # [B, 1, T, N]
        x = self.input_proj(x)                  # [B, d_short, T, N]

        for conv in self.temporal_convs:
            x = x + conv(x)

        x = self.spatial_conv(x)                # [B, d_short, T, N]

        # LayerNorm，取时间维最后一步
        x = x[:, :, -1, :]                      # [B, d_short, N]
        x = x.permute(0, 2, 1)                  # [B, N, d_short]
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# LongTermEncoder  —  简化 MTGNN（扩张因果卷积 + 自适应 GCN）
# ─────────────────────────────────────────────────────────────────────────────

class DilatedInceptionConv(nn.Module):
    """多尺度扩张卷积（替代 MTGNN 的 dilated_inception 模块）。"""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        # 三种核尺度：1, 3, 5
        ch = max(1, out_ch // 3)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, ch, kernel_size=(1, k),
                      padding=(0, (k - 1) * dilation // 2),
                      dilation=(1, dilation))
            for k in [1, 3, 5]
        ])
        # 拼接后投影到 out_ch
        self.proj = nn.Conv2d(ch * 3, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, N, T]"""
        return self.proj(torch.cat([c(x) for c in self.convs], dim=1))


class ASTERLongEncoder(nn.Module):
    """
    ASTER 长期编码器（简化 MTGNN）。

    原版 LongTermEncoder 采用 MTGNN（graph_constructor + mixprop +
    dilated_inception）。这里用等效的轻量实现：
      - 多层扩张因果卷积（DilatedInceptionConv）
      - 自适应邻接矩阵图传播

    Input:  [B, T, N, C]
    Output: [B, N, d_long]
    """

    def __init__(self, T: int, input_dim: int, num_nodes: int,
                 d_long: int = 32, num_layers: int = 3,
                 dropout: float = 0.3, node_dim: int = 16):
        super().__init__()

        self.start_conv = nn.Conv2d(input_dim, d_long, kernel_size=1)

        # 扩张卷积层（dilation = 1, 2, 4）
        self.filter_convs = nn.ModuleList()
        self.gate_convs   = nn.ModuleList()
        self.res_convs    = nn.ModuleList()
        dilations = [1, 2, 4][:num_layers]
        for d in dilations:
            self.filter_convs.append(DilatedInceptionConv(d_long, d_long, dilation=d))
            self.gate_convs.append(DilatedInceptionConv(d_long, d_long, dilation=d))
            self.res_convs.append(nn.Conv2d(d_long, d_long, kernel_size=1))

        # 自适应邻接矩阵（per-layer）
        self.E1 = nn.Parameter(torch.randn(num_layers, num_nodes, node_dim))
        self.E2 = nn.Parameter(torch.randn(num_layers, num_nodes, node_dim))
        self.gcn_W = nn.ModuleList([
            nn.Linear(d_long, d_long, bias=False) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, N, C]  →  [B, N, d_long]"""
        B, T, N, C = x.shape

        # [B, C, N, T]
        h = self.start_conv(x.permute(0, 3, 2, 1))   # [B, d_long, N, T]

        for i, (fc, gc, rc, W, e1, e2) in enumerate(
            zip(self.filter_convs, self.gate_convs,
                self.res_convs, self.gcn_W, self.E1, self.E2)
        ):
            # ── 时间方向（扩张卷积 GLU）──────────────────────────────────
            res = rc(h)
            f   = torch.tanh(fc(h))
            g   = torch.sigmoid(gc(h))
            h_t = self.dropout(f * g)            # [B, d_long, N, T']

            # ── 空间方向（自适应 GCN，在时间最后步）────────────────────
            h_last = h_t[:, :, :, -1]            # [B, d_long, N]
            A = F.softmax(F.relu(e1 @ e2.T), dim=-1)  # [N, N]
            # [B, N, d_long] → 图传播
            h_sp = W(h_last.permute(0, 2, 1))    # [B, N, d_long]
            h_sp = torch.einsum('nm,bmc->bnc', A, h_sp)  # [B, N, d_long]

            # 残差
            h = h_t + res                        # [B, d_long, N, T']

        # 取时间末尾步 → [B, N, d_long]
        h_out = h[:, :, :, -1].permute(0, 2, 1)  # [B, N, d_long]
        # 与图传播结果融合
        h_out = h_out + h_sp
        return self.norm(h_out)


# ─────────────────────────────────────────────────────────────────────────────
# ASTER Predictor
# ─────────────────────────────────────────────────────────────────────────────

class ASTERPredictor(EncoderResourcePredictor):
    """
    ASTER 基线（简化版）。

    短期：ASTERShortEncoder（Conv2D 残差）
    长期：ASTERLongEncoder（扩张卷积 + 自适应 GCN）
    后端：与主模型相同的 UnifiedResourceModule + MultiStepDecoder
    """

    def __init__(self, cfg: dict):
        N       = cfg["num_nodes"]
        D       = cfg["model_dim"]
        C       = cfg["input_dim"]
        dropout = cfg["dropout"]

        d_short = max(16, D // 2)
        d_long  = D

        short_enc = ASTERShortEncoder(
            T=cfg["T_short"], input_dim=C, num_nodes=N,
            d_short=d_short,
        )
        long_enc = ASTERLongEncoder(
            T=cfg["T_long"], input_dim=C, num_nodes=N,
            d_long=d_long, dropout=dropout,
        )

        # 短 / 长编码器输出维度可能不同，用公共 enc_dim=D 需要两个不同 proj
        # 这里通过让 short_enc 输出也 map 到 D 来统一
        # 做一个包装：补一层 Linear 让 short_enc 输出 D 维
        class _ShortWrap(nn.Module):
            def __init__(self, enc, d_in, d_out):
                super().__init__()
                self.enc = enc
                self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
            def forward(self, x):
                return self.proj(self.enc(x))

        short_enc_wrapped = _ShortWrap(short_enc, d_short, D)

        super().__init__(cfg, enc_dim=D,
                         short_encoder=short_enc_wrapped,
                         long_encoder=long_enc)
