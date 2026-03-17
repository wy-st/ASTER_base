# =============================================================================
# baselines/fc_lstm.py  ——  FC-LSTM 基线
# =============================================================================
# 参考：Yao et al. (2018) "Deep Multi-View Spatial-Temporal Network for
#       Taxi Demand Prediction" / Sutskever et al. LSTM 序列建模。
#
# 结构：
#   输入 [B, T, N, C]
#   → 每个节点独立走 FC + LSTM
#   → 取最后隐状态 [B, N, model_dim]
#   → EncoderResourcePredictor（共用资源融合 + 解码器 + DQN 智能体）
# =============================================================================

import torch
import torch.nn as nn
from .base import EncoderResourcePredictor


# ─────────────────────────────────────────────────────────────────────────────
# FC-LSTM 编码器
# ─────────────────────────────────────────────────────────────────────────────

class FCLSTMEncoder(nn.Module):
    """
    FC-LSTM 编码器：每个节点独立做 LSTM 时序建模。

    Input:  [B, T, N, C]
    Output: [B, N, model_dim]
    """

    def __init__(self, T: int, input_dim: int, num_nodes: int,
                 model_dim: int, dropout: float = 0.1):
        super().__init__()
        self.T = T
        self.N = num_nodes

        # 输入线性投影（per-timestep-per-node）
        self.input_fc = nn.Linear(input_dim, model_dim)

        # LSTM（按节点展开到 batch 维）
        self.lstm = nn.LSTM(
            input_size=model_dim,
            hidden_size=model_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, N, C]  →  [B, N, model_dim]
        """
        B, T, N, C = x.shape

        # 把节点维合并进 batch：[B*N, T, C]
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, C)

        # FC 投影
        x = self.dropout(torch.relu(self.input_fc(x)))  # [B*N, T, model_dim]

        # LSTM
        out, (h_n, _) = self.lstm(x)   # h_n: [2, B*N, model_dim]
        h = h_n[-1]                    # 最后一层末态 [B*N, model_dim]
        h = self.layer_norm(h)

        return h.view(B, N, -1)        # [B, N, model_dim]


# ─────────────────────────────────────────────────────────────────────────────
# FC-LSTM 预测器（完整 baseline 模型）
# ─────────────────────────────────────────────────────────────────────────────

class FCLSTMPredictor(EncoderResourcePredictor):
    """
    FC-LSTM baseline。

    短/长期窗口各用一个独立的 FCLSTMEncoder，
    后端复用主模型的资源感知融合 + 解码器。
    """

    def __init__(self, cfg: dict):
        D       = cfg["model_dim"]
        dropout = cfg["dropout"]

        short_enc = FCLSTMEncoder(
            T=cfg["T_short"], input_dim=cfg["input_dim"],
            num_nodes=cfg["num_nodes"], model_dim=D, dropout=dropout,
        )
        long_enc = FCLSTMEncoder(
            T=cfg["T_long"], input_dim=cfg["input_dim"],
            num_nodes=cfg["num_nodes"], model_dim=D, dropout=dropout,
        )

        super().__init__(cfg, enc_dim=D,
                         short_encoder=short_enc, long_encoder=long_enc)
