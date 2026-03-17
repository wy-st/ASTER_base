# =============================================================================
# baselines/staeformer.py  ——  STAEformer 基线
# =============================================================================
# 参考：Liu et al. (2023) "STAEformer: Spatio-Temporal Adaptive Embedding
#       Makes Vanilla Transformer Powerful for Traffic Forecasting"
#       (CIKM 2023, arXiv:2308.10425)
#
# 核心创新：可学习的空间自适应嵌入（per-node）+ 时间自适应嵌入（per-step），
# 加到输入特征上后，标准 Transformer 即可有效建模时空依赖。
#
# 结构：
#   输入 [B, T, N, C]
#   → input_proj: [B, T, N, d]
#   → + spatial_emb[N, d] + temporal_emb[T, d]  （广播相加）
#   → per-node 时间 Transformer: [B*N, T, d] → [B*N, T, d]
#   → per-timestep 节点 Transformer: [B*T, N, d] → [B*T, N, d]
#   → 取最后时间步 [B, N, d]  →  [B, N, model_dim]
#   → EncoderResourcePredictor
# =============================================================================

import torch
import torch.nn as nn
from .base import EncoderResourcePredictor


# ─────────────────────────────────────────────────────────────────────────────
# STAEformer 编码器
# ─────────────────────────────────────────────────────────────────────────────

class STAEformerEncoder(nn.Module):
    """
    STAEformer 编码器。

    Input:  [B, T, N, C]
    Output: [B, N, model_dim]
    """

    def __init__(self, T: int, input_dim: int, num_nodes: int,
                 model_dim: int, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.T = T
        self.N = num_nodes
        self.d = model_dim

        # ── 输入投影 ──────────────────────────────────────────────────────
        self.input_proj = nn.Linear(input_dim, model_dim)

        # ── 自适应嵌入（论文核心）────────────────────────────────────────
        # 空间自适应嵌入：每个节点独立的 d 维嵌入
        self.spatial_emb  = nn.Parameter(torch.randn(num_nodes, model_dim))
        # 时间自适应嵌入：每个时间步独立的 d 维嵌入
        self.temporal_emb = nn.Parameter(torch.randn(T, model_dim))

        # ── 时间轴 Transformer（per-node，在 T 维做注意力）───────────────
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer, num_layers=num_layers
        )

        # ── 空间轴 Transformer（per-timestep，在 N 维做注意力）──────────
        # 对大图（N≥1000），使用更少头以控制内存
        heads_spatial = min(num_heads, max(1, model_dim // 16))
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=heads_spatial,
            dim_feedforward=model_dim * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(
            spatial_layer, num_layers=1
        )

        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, N, C]  →  [B, N, model_dim]
        """
        B, T, N, C = x.shape

        # ① 输入投影
        h = self.input_proj(x)   # [B, T, N, d]

        # ② 加空间 & 时间自适应嵌入
        h = h + self.spatial_emb.view(1, 1, N, self.d)     # broadcast B, T
        h = h + self.temporal_emb.view(1, T, 1, self.d)    # broadcast B, N

        # ③ 时间轴 Transformer：每个节点独立地在 T 维做自注意力
        #    reshape → [B*N, T, d]
        h_t = h.permute(0, 2, 1, 3).reshape(B * N, T, self.d)
        h_t = self.temporal_transformer(h_t)                 # [B*N, T, d]
        h_t = h_t.reshape(B, N, T, self.d).permute(0, 2, 1, 3)  # [B, T, N, d]

        # ④ 空间轴 Transformer：在最后一个时间步对 N 节点做注意力
        #    取最后时间步 → [B, N, d]（减少内存：仅最后步参与空间注意力）
        h_last = h_t[:, -1, :, :]                            # [B, N, d]
        h_last = self.spatial_transformer(h_last)             # [B, N, d]

        return self.output_norm(h_last)                       # [B, N, d]


# ─────────────────────────────────────────────────────────────────────────────
# STAEformer Predictor
# ─────────────────────────────────────────────────────────────────────────────

class STAEformerPredictor(EncoderResourcePredictor):
    """
    STAEformer baseline。

    短/长期窗口各用一个独立的 STAEformerEncoder，
    后端复用主模型的资源感知融合 + 解码器。
    """

    def __init__(self, cfg: dict):
        D       = cfg["model_dim"]
        N       = cfg["num_nodes"]
        heads   = cfg["num_heads"]
        layers  = cfg["num_layers"]
        dropout = cfg["dropout"]
        C       = cfg["input_dim"]

        short_enc = STAEformerEncoder(
            T=cfg["T_short"], input_dim=C, num_nodes=N,
            model_dim=D, num_heads=heads, num_layers=layers, dropout=dropout,
        )
        long_enc = STAEformerEncoder(
            T=cfg["T_long"], input_dim=C, num_nodes=N,
            model_dim=D, num_heads=heads, num_layers=layers, dropout=dropout,
        )

        super().__init__(cfg, enc_dim=D,
                         short_encoder=short_enc, long_encoder=long_enc)
