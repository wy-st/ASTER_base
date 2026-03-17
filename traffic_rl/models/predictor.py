# =============================================================================
# models/predictor.py  ——  资源感知预测模型（来自 ASTER）
# =============================================================================
# 包含：
#   UnifiedResourceModule  —  按资源比例融合短/长期特征，决定预测步数 k
#   MultiStepDecoder       —  Transformer 解码器，生成 k 步事件概率
#   ResourcePredictor      —  整合编码器 + 融合模块 + 解码器
# =============================================================================

import torch
import torch.nn as nn

from models.encoder import TrafficEncoder


# ─────────────────────────────────────────────────────────────────────────────
# UnifiedResourceModule
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedResourceModule(nn.Module):
    """
    资源感知融合模块（ASTER 核心思想）。

    资源越充足 → 越多依赖长期特征 → 预测更长远的步数 k。
    资源越匮乏 → 越多依赖短期特征 → 预测较近的步数 k。

    Input:
        short_feat         : [B, N, C_common]
        long_feat          : [B, N, C_common]
        available_resources: int 或 [B] tensor

    Output:
        fused : [B, N, C_common]
        k     : [B] int tensor（每个样本的调度步数）
    """

    def __init__(self, C_common, num_nodes, total_resources,
                 K_max=12, num_heads=4, dropout=0.1):
        super().__init__()
        self.total_resources = total_resources
        self.K_max           = K_max

        # 节点间自注意力（融合后做空间交互）
        self.attn = nn.MultiheadAttention(
            C_common, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn  = nn.Sequential(
            nn.Linear(C_common, C_common * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(C_common * 4, C_common),
        )
        self.ln1  = nn.LayerNorm(C_common)
        self.ln2  = nn.LayerNorm(C_common)
        self.drop = nn.Dropout(dropout)

    def forward(self, short_feat, long_feat, available_resources):
        B, N, C = short_feat.shape
        device   = short_feat.device

        # 计算资源占比 ratio ∈ (0, 1]
        if isinstance(available_resources, (int, float)):
            ratio = torch.full((B,), available_resources / self.total_resources,
                               device=device, dtype=torch.float32)
        else:
            ratio = torch.as_tensor(available_resources, dtype=torch.float32,
                                    device=device) / self.total_resources
        ratio = ratio.clamp(0.001, 1.0).view(B, 1, 1)  # [B, 1, 1]

        # 加权融合：资源多 → 长期权重大
        fused = (1 - ratio) * short_feat + ratio * long_feat   # [B, N, C]

        # 节点间自注意力
        attn_out, _ = self.attn(fused, fused, fused)
        fused = self.ln1(fused + self.drop(attn_out))

        # FFN
        fused = self.ln2(fused + self.drop(self.ffn(fused)))

        # 根据资源比决定预测步数 k
        k = (ratio.squeeze() * self.K_max).round().clamp(min=1, max=self.K_max).long()
        if k.dim() == 0:
            k = k.unsqueeze(0)   # 保证 [B]

        return fused, k


# ─────────────────────────────────────────────────────────────────────────────
# MultiStepDecoder
# ─────────────────────────────────────────────────────────────────────────────

class MultiStepDecoder(nn.Module):
    """
    多步 Transformer 解码器（ASTER 风格）。

    给定融合后的记忆特征 memory [B, N, C]，
    自回归地生成 k 步的每节点事件概率（0~1）。

    Output: [B, k, N, 1]
    """

    def __init__(self, C_common, K_max=12, num_heads=4, dropout=0.1):
        super().__init__()
        self.K_max = K_max

        # 步骤位置嵌入（step 0 ~ K_max）
        self.step_embed = nn.Embedding(K_max + 1, C_common)

        # Transformer 解码器
        dec_layer = nn.TransformerDecoderLayer(
            d_model=C_common, nhead=num_heads,
            dim_feedforward=C_common * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=2)

        # 输出投影：每个解码步 → 每节点的事件概率
        self.out_proj = nn.Linear(C_common, 1)

    def forward(self, memory, k):
        """
        memory : [B, N, C_common]
        k      : int，本次解码步数

        返回: [B, k, N, 1]
        """
        B, N, C = memory.shape
        device   = memory.device

        # 步骤 query：[B, k, C]
        step_idx = torch.arange(k, device=device)
        tgt      = self.step_embed(step_idx).unsqueeze(0).expand(B, -1, -1)

        # 因果掩码（step i 只能看到 0..i-1）
        causal_mask = torch.triu(
            torch.full((k, k), float('-inf'), device=device), diagonal=1
        )

        # 解码：以节点特征为 memory，步骤嵌入为 query
        dec_out = self.decoder(tgt, memory, tgt_mask=causal_mask)  # [B, k, C]

        # 投影并广播到所有节点：[B, k, 1] → [B, k, N, 1]
        out = torch.sigmoid(self.out_proj(dec_out))                 # [B, k, 1]
        out = out.unsqueeze(2).expand(B, k, N, 1)                  # [B, k, N, 1]

        return out


# ─────────────────────────────────────────────────────────────────────────────
# ResourcePredictor
# ─────────────────────────────────────────────────────────────────────────────

class ResourcePredictor(nn.Module):
    """
    完整预测模型：

      short_term → short_encoder → proj_short ─┐
                                                ├─ UnifiedResourceModule → fused
      long_term  → long_encoder  → proj_long  ─┘
                                                └─ MultiStepDecoder → predictions

    forward 返回：
        predictions : [B, k_max, N, 1]   事件概率
        hidden      : [B, N, C_common]   融合特征（用于 DQN 状态）
        k           : [B]                每个样本的调度步数
    """

    def __init__(self, cfg):
        super().__init__()
        N         = cfg["num_nodes"]
        T_short   = cfg["T_short"]
        T_long    = cfg["T_long"]
        C         = cfg["input_dim"]
        D         = cfg["model_dim"]
        c_dim     = cfg["c_dim"]
        C_common  = cfg["C_common"]
        K_max     = cfg["K_max"]
        heads     = cfg["num_heads"]
        layers    = cfg["num_layers"]
        dropout   = cfg["dropout"]
        total_res = cfg["total_resources"]

        # 短期和长期编码器（T 不同，需要两个独立实例）
        self.short_encoder = TrafficEncoder(
            T=T_short, input_dim=C, num_nodes=N,
            model_dim=D, c_dim=c_dim,
            num_heads=heads, num_layers=layers, dropout=dropout,
        )
        self.long_encoder = TrafficEncoder(
            T=T_long, input_dim=C, num_nodes=N,
            model_dim=D, c_dim=c_dim,
            num_heads=heads, num_layers=layers, dropout=dropout,
        )

        # 投影到公共维度
        self.proj_short = nn.Linear(D, C_common)
        self.proj_long  = nn.Linear(D, C_common)

        # 资源感知融合 + 步数预测
        self.resource_module = UnifiedResourceModule(
            C_common=C_common, num_nodes=N,
            total_resources=total_res, K_max=K_max,
            num_heads=heads, dropout=dropout,
        )

        # 多步解码器
        self.decoder = MultiStepDecoder(
            C_common=C_common, K_max=K_max,
            num_heads=heads, dropout=dropout,
        )

    def forward(self, short_term, long_term, available_resources):
        """
        short_term         : [B, T_short, N, C]
        long_term          : [B, T_long,  N, C]
        available_resources: int 或 [B] tensor

        返回:
            predictions : [B, k_max, N, 1]
            hidden      : [B, N, C_common]
            k           : [B] int tensor
        """
        h_short = self.proj_short(self.short_encoder(short_term))  # [B, N, C_common]
        h_long  = self.proj_long(self.long_encoder(long_term))     # [B, N, C_common]

        fused, k = self.resource_module(h_short, h_long, available_resources)

        k_max       = int(k.max().item())
        predictions = self.decoder(fused, k_max)   # [B, k_max, N, 1]

        return predictions, fused, k
