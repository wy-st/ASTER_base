# =============================================================================
# models/hyperion.py  ——  完整模型（所有 Stage 在一个文件内）
# =============================================================================
#
# 流水线总览：
#
#   输入 short_term [B, T_short, N, F]
#        long_term  [B, T_long,  N, F]
#   │
#   ├─ Stage 1-4 : input_proj + AdaptiveGCN
#   │              → h_short [B,N,D]，h_long [B,N,D]
#   │              节点条件 c [B,N,c_dim]
#   │
#   ├─ Stage 2   : ResilienceEncoder
#   │              → τ [N]（弹性时间尺度），δ [N]（恢复率）
#   │
#   ├─ Stage 3   : HypergraphDiffusion（K 步迭代）
#   │              → x_diff [B,N,1]（事件扩散概率）
#   │
#   ├─ Stage 5   : 条件增强
#   │              c_aug = [c ‖ log_τ ‖ δ ‖ x_diff]  →  [B,N,c_dim+3]
#   │
#   ├─ Stage 6   : RC-GLN 调制注意力层 × L
#   │              → h_long, h_short  [B,N,D]
#   │              辅助损失：L_diff = MSE(x_diff, event_target)（在外部计算）
#   │
#   ├─ Stage 7   : proj → UnifiedResourceModule
#   │              → fused [B,N,C_common]，k [B]
#   │              → MultiStepDecoder → predictions [B,k_max,N,1]
#   │
#   ├─ Stage 8-9 : state_hidden = [h_long ‖ x_diff ‖ log_τ ‖ δ]
#   │              → [B, N, D+3]（DQN 状态由 train.py 拼接 res+cool+xy）
#   │
#   └─ 返回      : (predictions, state_hidden, k)
#                   self.last_x_diff 供 train_epoch.py 计算 L_diff
#
# 关键超参（均来自 cfg，可在 config.py 中调整）：
#   model_dim  D     — 编码器隐层维度
#   c_dim            — 节点条件嵌入维度
#   C_common         — 资源模块公共维度
#   num_heads        — 多头注意力头数
#   num_layers  L    — GLN 层数
#   K_max            — 最大预测步数
#   diff_K           — 超图扩散迭代步数（默认 3）
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _modulate(x, shift, scale):
    """GLN 自适应缩放：x * (1 + scale) + shift"""
    return x * (1.0 + scale) + shift


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1-4  :  自适应图卷积（Adaptive GCN）
# ─────────────────────────────────────────────────────────────────────────────

class _AdaptiveGCN(nn.Module):
    """
    混合传播自适应图卷积（参考 GWNet / MTGNN 的 mix-hop propagation）：
      A = softmax(relu(E1 @ E2))    [N, N]，可学习邻接矩阵
      output = Linear([x, Ax, A²x, ...])

    无需预计算图结构，所有数据集通用。
    """

    def __init__(self, num_nodes: int, model_dim: int,
                 order: int = 2, embed_dim: int = 10):
        super().__init__()
        self.order     = order
        self.nodevec1  = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.nodevec2  = nn.Parameter(torch.randn(embed_dim, num_nodes))
        self.lin       = nn.Linear(model_dim * (order + 1), model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, N, D]  →  [B, N, D]"""
        A = torch.softmax(
            torch.relu(self.nodevec1 @ self.nodevec2), dim=-1
        )  # [N, N]

        supports = [x]
        h = x
        for _ in range(self.order):
            h = torch.einsum('nm,bmd->bnd', A, h)
            supports.append(h)

        return self.lin(torch.cat(supports, dim=-1))   # [B, N, D]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2  :  弹性编码器（Resilience Encoder）
# ─────────────────────────────────────────────────────────────────────────────

class _ResilienceEncoder(nn.Module):
    """
    每节点可学习弹性参数（全局，不依赖输入）：
      τ   ∈ (0, ∞)  —  弹性时间尺度，越大表示系统恢复越慢
      δ   ∈ (0, 1)  —  恢复率（扩散步长），越大表示事件扩散越快

    使用 softplus / sigmoid 保证参数范围合法。
    """

    def __init__(self, num_nodes: int):
        super().__init__()
        # 参数在 unconstrained 空间存储，前向时转换
        self._raw_tau   = nn.Parameter(torch.zeros(num_nodes))
        self._raw_delta = nn.Parameter(torch.zeros(num_nodes))

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self._raw_tau)          # [N], >0

    @property
    def delta(self) -> torch.Tensor:
        return torch.sigmoid(self._raw_delta)     # [N], (0,1)

    def forward(self):
        """返回 (τ [N], δ [N])"""
        return self.tau, self.delta


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3  :  超图反应扩散（Hypergraph Reaction-Diffusion）
# ─────────────────────────────────────────────────────────────────────────────

class _HypergraphDiffusion(nn.Module):
    """
    超图反应扩散，K 步迭代：
      Θ = softmax(H) @ softmax(H).T      [N, N]  归一化超图传播算子
      x_{t+1} = (1 - δ) * x_t + δ * (Θ @ x_t)

    H ∈ R^{N × E}：可学习节点-超边关联矩阵（E = N//8 超边）。
    以速度通道（ch0 最后一步）作为初始扩散信号，
    输出每节点的事件扩散概率 x_diff ∈ (0,1)。
    """

    def __init__(self, num_nodes: int,
                 num_hyperedges: int = None, K: int = 3):
        super().__init__()
        self.K = K
        E = num_hyperedges or max(4, num_nodes // 8)
        # 可学习超边关联（小初始化避免 softmax 过早饱和）
        self.H = nn.Parameter(torch.randn(num_nodes, E) * 0.1)

    def forward(self, x_speed: torch.Tensor,
                delta: torch.Tensor) -> torch.Tensor:
        """
        x_speed : [B, N, 1]  —  速度特征（初始信号）
        delta   : [N]        —  每节点恢复率（扩散步长）
        返回    : [B, N, 1]  —  扩散后事件概率（sigmoid 输出）
        """
        H_norm = torch.softmax(self.H, dim=0)        # [N, E]
        Theta  = H_norm @ H_norm.t()                 # [N, N]

        x       = x_speed                            # [B, N, 1]
        delta_v = delta.view(1, -1, 1)               # [1, N, 1]

        for _ in range(self.K):
            diffuse = torch.einsum('nm,bm1->bn1', Theta, x)
            x = (1.0 - delta_v) * x + delta_v * diffuse

        return torch.sigmoid(x)   # [B, N, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5-6  :  RC-GLN 调制注意力层
# ─────────────────────────────────────────────────────────────────────────────

class _RCGLNLayer(nn.Module):
    """
    Resilience-Conditioned GLN（RC-GLN）调制 Transformer 块。

    与基础 SelfAttentionLayer 结构相同，但条件向量 c_aug 已包含
    log_τ / δ / x_diff，因此捕获了节点的弹性信息。

    GLN 把 c_aug 映射为 6 组调制参数：
      (shift_a, scale_a, gate_a) 作用在自注意力子层
      (shift_f, scale_f, gate_f) 作用在 FFN 子层

    Input:
        x     : [B, N, D]
        c_aug : [B, N, c_aug_dim]   (c_dim + 3)
    Output:
        x     : [B, N, D]
    """

    def __init__(self, model_dim: int, c_aug_dim: int,
                 ffn_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln1  = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.ln2  = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.ffn  = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
        )
        self.drop = nn.Dropout(dropout)
        # GLN 映射：c_aug → 6 × model_dim 调制参数
        self.gln  = nn.Sequential(
            nn.ReLU(),
            nn.Linear(c_aug_dim, 6 * model_dim),
        )

    def forward(self, x: torch.Tensor, c_aug: torch.Tensor) -> torch.Tensor:
        params = self.gln(c_aug)    # [B, N, 6*D]
        shift_a, scale_a, gate_a, \
        shift_f, scale_f, gate_f = params.chunk(6, dim=-1)

        # —— 自注意力子层 ——
        x_mod = _modulate(self.ln1(x), shift_a, scale_a)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + self.drop(gate_a * attn_out)

        # —— FFN 子层 ——
        x_mod = _modulate(self.ln2(x), shift_f, scale_f)
        x = x + self.drop(gate_f * self.ffn(x_mod))

        return x


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7  :  资源感知融合模块（UnifiedResourceModule）
# ─────────────────────────────────────────────────────────────────────────────

class _UnifiedResourceModule(nn.Module):
    """
    资源比例加权融合短/长期特征，决定预测步数 k：
      ratio  = available / total_resources  ∈ (0, 1]
      fused  = (1-ratio) * short + ratio * long
      fused  = 节点间自注意力 + FFN（Transformer 风格精炼）
      k      = round(ratio * K_max)，确保 k ∈ [1, K_max]
    """

    def __init__(self, C_common: int, total_resources: int,
                 K_max: int, num_heads: int, dropout: float):
        super().__init__()
        self.total_resources = total_resources
        self.K_max           = K_max

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

        if isinstance(available_resources, (int, float)):
            ratio = torch.full(
                (B,), available_resources / self.total_resources,
                device=device, dtype=torch.float32,
            )
        else:
            ratio = torch.as_tensor(
                available_resources, dtype=torch.float32, device=device
            ) / self.total_resources
        ratio = ratio.clamp(0.001, 1.0).view(B, 1, 1)   # [B, 1, 1]

        fused = (1.0 - ratio) * short_feat + ratio * long_feat

        attn_out, _ = self.attn(fused, fused, fused)
        fused = self.ln1(fused + self.drop(attn_out))
        fused = self.ln2(fused + self.drop(self.ffn(fused)))

        k = (ratio.squeeze() * self.K_max).round().clamp(min=1, max=self.K_max).long()
        if k.dim() == 0:
            k = k.unsqueeze(0)   # 保证 [B]

        return fused, k   # ([B,N,C], [B])


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7  :  多步 Transformer 解码器（MultiStepDecoder）
# ─────────────────────────────────────────────────────────────────────────────

class _MultiStepDecoder(nn.Module):
    """
    给定融合特征 memory [B, N, C]，
    自回归生成 k 步的每节点事件概率，输出 [B, k, N, 1]。

    步骤位置嵌入作为 query，融合特征作为 memory（Cross-Attention）。
    使用因果掩码保证 step-i 只看 0..i-1。
    """

    def __init__(self, C_common: int, K_max: int,
                 num_heads: int, dropout: float):
        super().__init__()
        self.K_max      = K_max
        self.step_embed = nn.Embedding(K_max + 1, C_common)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=C_common,
            nhead=num_heads,
            dim_feedforward=C_common * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN，训练更稳定
        )
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=2)
        self.out_proj = nn.Linear(C_common, 1)

    def forward(self, memory: torch.Tensor, k: int) -> torch.Tensor:
        """
        memory : [B, N, C_common]
        k      : int，本次解码步数

        返回   : [B, k, N, 1]
        """
        B, N, C = memory.shape
        device   = memory.device

        step_idx = torch.arange(k, device=device)
        tgt      = self.step_embed(step_idx).unsqueeze(0).expand(B, -1, -1)  # [B,k,C]

        causal_mask = torch.triu(
            torch.full((k, k), float('-inf'), device=device), diagonal=1
        )

        dec_out = self.decoder(tgt, memory, tgt_mask=causal_mask)  # [B, k, C]
        out     = torch.sigmoid(self.out_proj(dec_out))            # [B, k, 1]
        return out.unsqueeze(2).expand(B, k, N, 1)                 # [B, k, N, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Hyperion  :  完整流水线（Stage 1-9 集成）
# ─────────────────────────────────────────────────────────────────────────────

class Hyperion(nn.Module):
    """
    完整资源感知交通事件预测模型（单文件实现）。

    与训练循环的接口：
        forward(short_term, long_term, available_resources)
          → (predictions [B,k_max,N,1],
             state_hidden [B,N,D+3],      # h_long ‖ x_diff ‖ log_τ ‖ δ
             k [B])

        self.last_x_diff : [B,N,1]
            在每次 forward 后更新，供 train_epoch.py 计算：
              L_diff = MSE(x_diff, event_target)
              L = L_pred + 0.1 * L_diff

    DQN 状态维度：
        state_dim = N * (state_hidden_dim + 4)
                  = N * (D + 3 + res(1) + cool(1) + x(1) + y(1))
                  = N * (D + 7)
        （其中 +4 由 construct_rl_state 在外部拼接）
    """

    def __init__(self, cfg: dict):
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
        diff_K    = cfg.get("diff_K", 3)

        # ── Stage 1 : 时间展平 + 线性投影（短/长期各一个）────────────
        self.short_proj = nn.Linear(T_short * C, D)
        self.long_proj  = nn.Linear(T_long  * C, D)

        # ── Stage 1-4 : 自适应图卷积（短/长期各一个）───────────────
        self.gcn_short = _AdaptiveGCN(N, D)
        self.gcn_long  = _AdaptiveGCN(N, D)

        # ── Stage 1-4 : 可学习节点条件嵌入 c ────────────────────────
        self.node_cond = nn.Parameter(torch.randn(N, c_dim))

        # ── Stage 2 : 弹性编码器 ─────────────────────────────────────
        self.resil_enc = _ResilienceEncoder(N)

        # ── Stage 3 : 超图反应扩散 ───────────────────────────────────
        self.hyper_diff = _HypergraphDiffusion(N, K=diff_K)

        # ── Stage 5-6 : RC-GLN 调制注意力层（c_aug_dim = c_dim + 3）─
        c_aug_dim = c_dim + 3   # c ‖ log_τ(1) ‖ δ(1) ‖ x_diff(1)

        self.rc_gln_long = nn.ModuleList([
            _RCGLNLayer(D, c_aug_dim, ffn_dim=D * 4,
                        num_heads=heads, dropout=dropout)
            for _ in range(layers)
        ])
        self.rc_gln_short = nn.ModuleList([
            _RCGLNLayer(D, c_aug_dim, ffn_dim=D * 4,
                        num_heads=heads, dropout=dropout)
            for _ in range(layers)
        ])

        # ── Stage 7 : 投影到公共维度 ────────────────────────────────
        self.proj_long  = nn.Linear(D, C_common)
        self.proj_short = nn.Linear(D, C_common)

        # ── Stage 7 : 资源感知融合 + 步数决定 ───────────────────────
        self.resource_module = _UnifiedResourceModule(
            C_common=C_common, total_resources=total_res,
            K_max=K_max, num_heads=heads, dropout=dropout,
        )

        # ── Stage 7 : 多步解码器 ─────────────────────────────────────
        self.decoder = _MultiStepDecoder(
            C_common=C_common, K_max=K_max,
            num_heads=heads, dropout=dropout,
        )

        # ── 对外暴露维度（供 train.py 计算 state_dim）───────────────
        # state_hidden = h_long[D] ‖ x_diff[1] ‖ log_τ[1] ‖ δ[1]
        self.state_hidden_dim: int = D + 3

        # 用于外部读取 x_diff 计算辅助损失
        self.last_x_diff: torch.Tensor = None

    # ─────────────────────────────────────────────────────────────────────────
    def _encode_path(
        self,
        x:          torch.Tensor,   # [B, T, N, C]
        proj:       nn.Module,      # Linear(T*C → D)
        gcn:        nn.Module,      # _AdaptiveGCN
        rc_layers:  nn.ModuleList,  # RC-GLN 层列表
        c_aug:      torch.Tensor,   # [B, N, c_aug_dim]
    ) -> torch.Tensor:              # [B, N, D]
        """
        通用编码路径（短期 / 长期共用）：
          [B,T,N,C] → flatten → proj [B,N,D] → GCN → RC-GLN×L → [B,N,D]
        """
        B, T, N, C = x.shape
        h = proj(x.permute(0, 2, 1, 3).reshape(B, N, T * C))  # [B, N, D]
        h = gcn(h)
        for layer in rc_layers:
            h = layer(h, c_aug)
        return h

    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        short_term:          torch.Tensor,           # [B, T_short, N, C]
        long_term:           torch.Tensor,           # [B, T_long,  N, C]
        available_resources,                         # int 或 [B] tensor
    ):
        """
        Stage 2-9 完整前向计算。

        返回：
            predictions  : [B, k_max, N, 1]
            state_hidden : [B, N, D+3]        ← h_long ‖ x_diff ‖ log_τ ‖ δ
            k            : [B] int tensor
        """
        B = long_term.size(0)
        N = long_term.size(2)

        # ── Stage 2 : 弹性参数 ────────────────────────────────────────
        tau, delta = self.resil_enc()                        # [N], [N]
        log_tau    = torch.log(tau + 1e-6)                   # [N]

        # ── Stage 3 : 超图反应扩散 ────────────────────────────────────
        # 用长期窗口最后一步的速度通道（ch0）作为初始扩散信号
        x_speed = long_term[:, -1, :, 0:1]                  # [B, N, 1]
        x_diff  = self.hyper_diff(x_speed, delta)            # [B, N, 1]
        self.last_x_diff = x_diff.detach()                   # 供外部 L_diff

        # ── Stage 4 : 节点条件嵌入 c（扩展到 batch）────────────────
        c = self.node_cond.unsqueeze(0).expand(B, -1, -1)   # [B, N, c_dim]

        # ── Stage 5 : c_aug = [c ‖ log_τ ‖ δ ‖ x_diff] ─────────────
        log_tau_exp = log_tau.view(1, N, 1).expand(B, -1, -1)  # [B, N, 1]
        delta_exp   = delta.view(1, N, 1).expand(B, -1, -1)    # [B, N, 1]
        c_aug = torch.cat(
            [c, log_tau_exp, delta_exp, x_diff], dim=-1
        )  # [B, N, c_dim+3]

        # ── Stage 6 : RC-GLN 编码（长/短期各走独立的 L 层）──────────
        h_long  = self._encode_path(
            long_term,  self.long_proj,  self.gcn_long,
            self.rc_gln_long,  c_aug,
        )   # [B, N, D]

        h_short = self._encode_path(
            short_term, self.short_proj, self.gcn_short,
            self.rc_gln_short, c_aug,
        )   # [B, N, D]

        # ── Stage 7 : 资源感知融合 + 步数决定 ────────────────────────
        h_long_c  = self.proj_long(h_long)    # [B, N, C_common]
        h_short_c = self.proj_short(h_short)  # [B, N, C_common]
        fused, k  = self.resource_module(h_short_c, h_long_c, available_resources)

        # ── 多步解码 ──────────────────────────────────────────────────
        k_max       = int(k.max().item())
        predictions = self.decoder(fused, k_max)   # [B, k_max, N, 1]

        # ── State 8 : 构建 DQN 隐层状态 ─────────────────────────────
        # h_long ‖ x_diff ‖ log_τ ‖ δ  →  [B, N, D+3]
        state_hidden = torch.cat(
            [h_long, x_diff, log_tau_exp, delta_exp], dim=-1
        )

        return predictions, state_hidden, k
