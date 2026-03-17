# =============================================================================
# baselines/base.py  ——  所有 baseline 共用的骨架
# =============================================================================
# EncoderResourcePredictor：
#   接收两个编码器（short_encoder / long_encoder），均输出 [B, N, D]，
#   再接 UnifiedResourceModule + MultiStepDecoder（与主模型完全相同），
#   返回 (predictions, fused, k) 以便直接接入现有训练循环。
#
# 各 baseline 只需实现自己的 Encoder，其余逻辑完全复用。
# =============================================================================

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.predictor import UnifiedResourceModule, MultiStepDecoder


class EncoderResourcePredictor(nn.Module):
    """
    通用 baseline 骨架。

    子类只需在 __init__ 里设置:
        self.short_encoder  —  [B, T_short, N, C] → [B, N, enc_dim]
        self.long_encoder   —  [B, T_long,  N, C] → [B, N, enc_dim]
        self.enc_dim        —  编码器输出维度

    然后调用 super().__init__(cfg) 即可。
    """

    def __init__(self, cfg, enc_dim: int, short_encoder: nn.Module, long_encoder: nn.Module):
        super().__init__()

        N        = cfg["num_nodes"]
        C_common = cfg["C_common"]
        K_max    = cfg["K_max"]
        heads    = cfg["num_heads"]
        dropout  = cfg["dropout"]
        total    = cfg["total_resources"]

        self.short_encoder = short_encoder
        self.long_encoder  = long_encoder

        # 投影到公共维度
        self.proj_short = nn.Linear(enc_dim, C_common)
        self.proj_long  = nn.Linear(enc_dim, C_common)

        # 资源感知融合
        self.resource_module = UnifiedResourceModule(
            C_common=C_common, num_nodes=N,
            total_resources=total, K_max=K_max,
            num_heads=heads, dropout=dropout,
        )
        # 多步解码
        self.decoder = MultiStepDecoder(
            C_common=C_common, K_max=K_max,
            num_heads=heads, dropout=dropout,
        )

    def forward(self, short_term, long_term, available_resources):
        """
        short_term : [B, T_short, N, C]
        long_term  : [B, T_long,  N, C]
        返回:
            predictions : [B, k_max, N, 1]
            fused       : [B, N, C_common]
            k           : [B]
        """
        h_s = self.proj_short(self.short_encoder(short_term))  # [B, N, C_common]
        h_l = self.proj_long(self.long_encoder(long_term))     # [B, N, C_common]

        fused, k = self.resource_module(h_s, h_l, available_resources)

        k_max       = int(k.max().item())
        predictions = self.decoder(fused, k_max)              # [B, k_max, N, 1]

        return predictions, fused, k
