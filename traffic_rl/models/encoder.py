# =============================================================================
# models/encoder.py  ——  时空编码器（来自 ConFormer）
# =============================================================================
# 包含：
#   SelfAttentionLayer  —  GLN 条件化的多头自注意力 + FFN 块
#   TrafficEncoder      —  输入 [B, T, N, C]，输出 [B, N, model_dim]
# =============================================================================

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _modulate(x, shift, scale):
    """GLN 自适应缩放：x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


# ─────────────────────────────────────────────────────────────────────────────
# SelfAttentionLayer
# ─────────────────────────────────────────────────────────────────────────────

class SelfAttentionLayer(nn.Module):
    """
    一个 GLN（Gated Linear Network）条件化的 Transformer 块。

    ConFormer 的核心创新：用外部条件信号 c 来动态调制 LayerNorm 的
    shift/scale 以及注意力 / FFN 的门控 gate。

    Input:
        x : [B, N, model_dim]   —— 特征
        c : [B, N, c_dim]       —— 条件信号（来自节点嵌入）
    Output:
        x : [B, N, model_dim]
    """

    def __init__(self, model_dim, c_dim, ffn_dim=256, num_heads=4, dropout=0.1):
        super().__init__()

        # 多头自注意力（在 N 个节点之间做）
        self.attn = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln1  = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.ln2  = nn.LayerNorm(model_dim, elementwise_affine=False)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
        )
        self.drop = nn.Dropout(dropout)

        # GLN：把条件 c 映射成 6 个向量（两组 shift/scale/gate）
        self.gln = nn.Sequential(
            nn.ReLU(),
            nn.Linear(c_dim, 6 * model_dim),
        )

    def forward(self, x, c):
        # 从条件信号里解出 6 个调制参数
        params = self.gln(c)   # [B, N, 6*model_dim]
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = params.chunk(6, dim=-1)

        # —— 自注意力子层 ——
        x_mod = _modulate(self.ln1(x), shift_a, scale_a)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + self.drop(gate_a * attn_out)

        # —— FFN 子层 ——
        x_mod = _modulate(self.ln2(x), shift_f, scale_f)
        x = x + self.drop(gate_f * self.ffn(x_mod))

        return x


# ─────────────────────────────────────────────────────────────────────────────
# TrafficEncoder
# ─────────────────────────────────────────────────────────────────────────────

class TrafficEncoder(nn.Module):
    """
    时空编码器（ConFormer 风格）。

    时间维处理：把 T 个时间步的特征 flatten 后线性投影到 model_dim。
    空间维处理：用 L 层 GLN 条件化注意力在 N 个节点之间建模空间依赖。
    条件信号  ：可学习的每节点嵌入向量（替代 ConFormer 里的 GCN 传播）。

    Input:  x [B, T, N, C]
    Output: h [B, N, model_dim]
    """

    def __init__(self, T, input_dim, num_nodes, model_dim, c_dim,
                 num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()

        # 时间 × 特征 → model_dim
        self.input_proj = nn.Linear(T * input_dim, model_dim)

        # 可学习的节点条件嵌入
        self.node_cond = nn.Parameter(torch.randn(num_nodes, c_dim))

        # 空间注意力层
        self.attn_layers = nn.ModuleList([
            SelfAttentionLayer(
                model_dim, c_dim,
                ffn_dim=model_dim * 4,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: [B, T, N, C]
        返回: [B, N, model_dim]
        """
        B, T, N, C = x.shape

        # [B, N, T*C] → [B, N, model_dim]
        h = self.input_proj(x.permute(0, 2, 1, 3).reshape(B, N, T * C))

        # 条件向量扩展到 batch 维：[B, N, c_dim]
        c = self.node_cond.unsqueeze(0).expand(B, -1, -1)

        for layer in self.attn_layers:
            h = layer(h, c)

        return h  # [B, N, model_dim]
