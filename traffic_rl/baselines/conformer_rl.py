# =============================================================================
# baselines/conformer_rl.py  ——  ConFormer + RL 基线
# =============================================================================
# 参考：wy-st/ConFormer_-base（ConFormer 论文官方代码）
#
# 前半部分（编码器）：ConFormer 原始架构
#   - 速度通道 ch0 → 线性投影 → input_embedding_dim
#   - ToD 通道 ch1 → Embedding 查表 → tod_embedding_dim
#   - DoW 通道 ch2 → Embedding 查表 → dow_embedding_dim
#   - 可学习节点嵌入（node_embedding_dim）
#   - 自适应图卷积（GCN with learned nodevec1/nodevec2）产生条件 c
#   - L 层 GLN-条件化自注意力（SelfAttentionLayer）
#
# 后半部分（RL 后端）：与主模型完全相同
#   - EncoderResourcePredictor（UnifiedResourceModule + MultiStepDecoder）
#   - DQNAgent（双 Q 网络，多目标优化）
#
# 输出接口：(predictions, fused, k)，与现有训练循环兼容。
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import EncoderResourcePredictor


# ─────────────────────────────────────────────────────────────────────────────
# ConFormer 自注意力层（GLN 条件化）—— 直接移植原论文代码
# ─────────────────────────────────────────────────────────────────────────────

def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


class _AttentionLayer(nn.Module):
    """多头自注意力（带 scaled_dot_product_attention）。"""

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D]"""
        B, N, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # 拆头 [B, N, H, head_dim] → [B, H, N, head_dim]
        def split_heads(t):
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        out = F.scaled_dot_product_attention(q, k, v)   # [B, H, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class _ConFormerSALayer(nn.Module):
    """
    ConFormer 核心块：GLN 条件化的 多头自注意力 + FFN。

    x: [B, N, model_dim]
    c: [B, N, c_dim]   （由 GCN 产生的条件信号）
    """

    def __init__(self, model_dim: int, c_dim: int,
                 ffn_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = _AttentionLayer(model_dim, num_heads)
        self.ln1  = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.ln2  = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.ffn  = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
        )
        self.drop = nn.Dropout(dropout)
        # GLN：c → 6 组调制参数
        self.gln  = nn.Sequential(nn.ReLU(), nn.Linear(c_dim, 6 * model_dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        p = self.gln(c)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = p.chunk(6, dim=-1)

        x_mod    = _modulate(self.ln1(x), shift_a, scale_a)
        x        = x + self.drop(gate_a * self.attn(x_mod))

        x_mod    = _modulate(self.ln2(x), shift_f, scale_f)
        x        = x + self.drop(gate_f * self.ffn(x_mod))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# GCN（自适应邻接矩阵）—— 移植自 ConFormer 原代码
# ─────────────────────────────────────────────────────────────────────────────

class _AdpGCN(nn.Module):
    """
    自适应图卷积（ConFormer GCN 模块）。
    不依赖外部邻接矩阵，通过可学习的节点向量构建动态图。
    c_in = model_dim - input_embedding_dim（即 condition 维度）
    """

    def __init__(self, c_dim: int, num_nodes: int,
                 gcn_depth: int = 2, dropout: float = 0.1,
                 alpha: float = 0.05):
        super().__init__()
        self.depth   = gcn_depth
        self.alpha   = alpha
        self.drop    = nn.Dropout(dropout)
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10))
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes))

        # mix-propagation 投影
        total_in = c_dim * (gcn_depth + 1)
        self.mlp = nn.Sequential(
            nn.Linear(total_in, c_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, c_dim]  →  [B, N, c_dim]"""
        A = F.softmax(
            F.relu(self.nodevec1 @ self.nodevec2), dim=-1
        )                               # [N, N]

        out = [x]
        h = x
        for _ in range(self.depth):
            # H' = (1 - α) A H + α H0
            h = (1 - self.alpha) * torch.einsum('nm,bnc->bmc', A, h) \
                + self.alpha * x
            h = self.drop(h)
            out.append(h)

        return self.mlp(torch.cat(out, dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# ConFormer 编码器（移植原论文 encoder 部分）
# ─────────────────────────────────────────────────────────────────────────────

class ConFormerEncoder(nn.Module):
    """
    ConFormer 编码器（精确复现原论文前半部分）。

    与我们的 TrafficEncoder 的主要区别：
      - 速度 ch0 通过 input_proj 线性投影
      - ToD  ch1 / DoW ch2 通过 Embedding 查表（原论文做法）
      - 条件信号 c 来自自适应 GCN 而非简单可学习嵌入

    Input:  [B, T, N, 3]   (速度 + ToD + DoW)
    Output: [B, N, model_dim]
    """

    def __init__(self, T: int, num_nodes: int,
                 input_embedding_dim: int = 32,
                 tod_embedding_dim:   int = 32,
                 dow_embedding_dim:   int = 32,
                 node_embedding_dim:  int = 32,
                 steps_per_day:       int = 144,
                 num_heads:           int = 4,
                 num_layers:          int = 3,
                 dropout:             float = 0.1):
        super().__init__()

        self.steps_per_day      = steps_per_day
        self.tod_embedding_dim  = tod_embedding_dim
        self.dow_embedding_dim  = dow_embedding_dim
        self.node_embedding_dim = node_embedding_dim

        model_dim = (input_embedding_dim
                     + tod_embedding_dim
                     + dow_embedding_dim
                     + node_embedding_dim)
        c_dim     = model_dim - input_embedding_dim

        # 速度投影（ch0 × T → input_embedding_dim）
        self.input_proj = nn.Linear(T, input_embedding_dim)   # T × 1 → D_in

        # 嵌入查表
        if tod_embedding_dim > 0:
            self.tod_emb  = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_emb  = nn.Embedding(7, dow_embedding_dim)
        if node_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(num_nodes, node_embedding_dim))
            )

        # 自适应 GCN 产生条件 c
        self.gcn = _AdpGCN(c_dim, num_nodes, gcn_depth=2, dropout=dropout)

        # GLN 条件化注意力层
        self.layers = nn.ModuleList([
            _ConFormerSALayer(model_dim, c_dim,
                              ffn_dim=model_dim * 4,
                              num_heads=num_heads,
                              dropout=dropout)
            for _ in range(num_layers)
        ])

        self._model_dim = model_dim

    @property
    def output_dim(self):
        return self._model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, N, 3]  → [B, N, model_dim]
        ch0=速度  ch1=ToD(0~1)  ch2=DoW(0~6)
        """
        B, T, N, _ = x.shape

        # ── 速度投影 ──────────────────────────────────────────────────────
        # x[..., 0]: [B, T, N] → permute → [B, N, T] → Linear → [B, N, D_in]
        speed = x[..., 0].permute(0, 2, 1)                     # [B, N, T]
        h_speed = self.input_proj(speed)                        # [B, N, D_in]

        features = []

        # ── ToD 嵌入 ─────────────────────────────────────────────────────
        if self.tod_embedding_dim > 0:
            tod_idx = (x[:, -1, :, 1] * self.steps_per_day).long().clamp(
                0, self.steps_per_day - 1
            )                                                   # [B, N]
            features.append(self.tod_emb(tod_idx))              # [B, N, D_tod]

        # ── DoW 嵌入 ─────────────────────────────────────────────────────
        if self.dow_embedding_dim > 0:
            dow_idx = x[:, -1, :, 2].long().clamp(0, 6)        # [B, N]
            features.append(self.dow_emb(dow_idx))              # [B, N, D_dow]

        # ── 节点嵌入 ─────────────────────────────────────────────────────
        if self.node_embedding_dim > 0:
            features.append(
                self.node_emb.unsqueeze(0).expand(B, -1, -1)   # [B, N, D_node]
            )

        # ── 拼接所有特征 ──────────────────────────────────────────────────
        if features:
            c_raw = torch.cat(features, dim=-1)                 # [B, N, c_dim]
        else:
            c_raw = torch.zeros(B, N, 0, device=x.device)

        # GCN 传播条件
        c = self.gcn(c_raw) if c_raw.shape[-1] > 0 else c_raw  # [B, N, c_dim]

        # 拼接 x = [speed_emb || conditions]
        h = torch.cat([h_speed, c_raw], dim=-1)                 # [B, N, model_dim]

        # GLN 条件化注意力
        for layer in self.layers:
            h = layer(h, c)

        return h    # [B, N, model_dim]


# ─────────────────────────────────────────────────────────────────────────────
# ConFormer+RL Predictor
# ─────────────────────────────────────────────────────────────────────────────

class ConFormerRLPredictor(EncoderResourcePredictor):
    """
    ConFormer 编码器 + ASTER RL 后端。

    前半（编码器）：ConFormer 原论文实现
      - 速度 → 线性投影
      - ToD/DoW → Embedding 查表
      - 自适应 GCN 条件 + GLN-SA 层
    后半（RL）：UnifiedResourceModule + MultiStepDecoder（与主模型完全相同）

    短/长期窗口各一个 ConFormerEncoder 实例。
    """

    def __init__(self, cfg: dict):
        N   = cfg["num_nodes"]
        spd = cfg["steps_per_day"]
        h   = cfg["num_heads"]
        L   = cfg["num_layers"]
        dr  = cfg["dropout"]

        # ── 编码器参数（与主模型 model_dim 对齐）────────────────────────
        # ConFormer model_dim = input_emb + tod_emb + dow_emb + node_emb
        # 这里均分 cfg["model_dim"] 给 4 个分量
        quarter  = cfg["model_dim"] // 4
        remainder = cfg["model_dim"] - quarter * 3

        short_enc = ConFormerEncoder(
            T=cfg["T_short"],  num_nodes=N,
            input_embedding_dim=remainder,
            tod_embedding_dim=quarter,
            dow_embedding_dim=quarter,
            node_embedding_dim=quarter,
            steps_per_day=spd,
            num_heads=h, num_layers=L, dropout=dr,
        )
        long_enc = ConFormerEncoder(
            T=cfg["T_long"], num_nodes=N,
            input_embedding_dim=remainder,
            tod_embedding_dim=quarter,
            dow_embedding_dim=quarter,
            node_embedding_dim=quarter,
            steps_per_day=spd,
            num_heads=h, num_layers=L, dropout=dr,
        )

        D = short_enc.output_dim   # = cfg["model_dim"]

        super().__init__(cfg, enc_dim=D,
                         short_encoder=short_enc, long_encoder=long_enc)
