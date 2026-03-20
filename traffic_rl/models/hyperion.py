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
#   k_per_edge       — 超图每条超边保留节点数（top-k 稀疏，默认 max(4, N//32)）
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
    全局弹性参数（标量，整个网络共享）：
      τ ∈ (0, ∞)  —  ODE 积分时域长度，越大扩散越深
      δ ∈ (0, 1)  —  扩散强度因子，越大事件传播越广

    两个参数在 unconstrained 空间存储，前向时映射到合法范围。
    """

    def __init__(self):
        super().__init__()
        self._raw_tau   = nn.Parameter(torch.zeros(1))   # softplus → τ
        self._raw_delta = nn.Parameter(torch.zeros(1))   # sigmoid  → δ

    def forward(self):
        """返回 (τ [1], δ [1])  全局标量"""
        tau   = F.softplus(self._raw_tau)      # (0, ∞)
        delta = torch.sigmoid(self._raw_delta) # (0, 1)
        return tau, delta


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3  :  超图卷积（Hypergraph Convolution，单次）
# ─────────────────────────────────────────────────────────────────────────────

class _GraphODEHypergraph(nn.Module):
    """
    Graph ODE → 稀疏超图卷积，一体化模块，整个网络只调用一次。

    Stage 1 — Graph ODE（连续扩散，Euler 离散）：
      dH/dt = tanh( A_ode @ H @ W_ode )
      A_ode = softmax(relu(nv1 @ nv2))   可学习图，不依赖固定拓扑
      从 H(0) = h_gcn 积分到 H(δ)，δ 为全局扩散深度标量
      → H_T [B, N, D]  扩散后的时空特征

    Stage 2 — 动态超边构造（MTGNN top-k 稀疏）：
      H_logit = edge_proj(H_T)    [B, N, E]
      对每条超边保留 top-k 最强节点 → H_sparse [B, N, E]
      扩散后特征相似的节点自然被划入同一超边

    Stage 3 — 归一化超图卷积（Feng et al. AAAI 2019）：
      a[b,e,d]   = Σ_n H_sparse[b,n,e] * H_T[b,n,d]   节点→超边
      out[b,n,d] = D_v^{-1} Σ_e H_w[b,n,e] * a[b,e,d]  超边→节点
      → Linear(D→1) + sigmoid → x_diff [B, N, 1]

    两步 einsum，不显式构造 [N, N] 矩阵。
    """

    def __init__(self, num_nodes: int, in_dim: int,
                 num_hyperedges: int = None,
                 k_per_edge:    int = None,
                 ode_steps:     int = 3,
                 embed_dim:     int = 10):
        super().__init__()
        E = num_hyperedges or max(4, num_nodes // 8)
        self.k_per_edge = k_per_edge or max(4, num_nodes // 32)
        self.ode_steps  = ode_steps

        # Stage 1：Graph ODE 可学习邻接（MTGNN 风格）
        self.nv1    = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.nv2    = nn.Parameter(torch.randn(embed_dim, num_nodes))
        # ODE 动力学线性变换 W_ode
        self.ode_W  = nn.Linear(in_dim, in_dim, bias=False)

        # Stage 2：超边分配投影
        self.edge_proj = nn.Linear(in_dim, E)

        # Stage 3：可学习超边权重
        self.edge_W = nn.Parameter(torch.ones(E))

    def _f(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """ODE 右端项: f(H) = tanh(A @ H @ W_ode)"""
        return torch.tanh(torch.einsum('nm,bmd->bnd', A, self.ode_W(H)))

    def forward(self, h: torch.Tensor,
                delta: torch.Tensor) -> torch.Tensor:
        """
        h     : [B, N, D]  —  GCN 输出（ODE 初始状态 H(0)）
        delta : [1]        —  全局扩散深度（ODE 积分时间 T = δ）
        返回  : [B, N, 1]  —  超图卷积后的事件传播概率 ∈ (0,1)
        """
        B, N, D = h.shape
        k = min(self.k_per_edge, N)

        # ── Stage 1 : Graph ODE（固定步数 Euler 积分）───────────────
        A  = torch.softmax(torch.relu(self.nv1 @ self.nv2), dim=-1)  # [N, N]
        dt = delta / self.ode_steps                                   # 标量步长
        H  = h
        for _ in range(self.ode_steps):
            H = H + dt * self._f(H, A)
        # H : [B, N, D]  即 H(δ)，扩散后的时空特征

        # ── Stage 2 : 动态超边构造 ───────────────────────────────────
        H_logit   = self.edge_proj(H)               # [B, N, E]
        H_t       = H_logit.permute(0, 2, 1)        # [B, E, N]
        topk_vals, topk_idx = H_t.topk(k, dim=-1)   # [B, E, k]
        sparse_w  = torch.softmax(topk_vals, dim=-1) # [B, E, k]
        H_sparse_t = torch.zeros_like(H_t)
        H_sparse_t.scatter_(-1, topk_idx, sparse_w)
        H_sparse  = H_sparse_t.permute(0, 2, 1)     # [B, N, E]

        # ── Stage 3 : 归一化超图卷积 ─────────────────────────────────
        D_v  = H_sparse.sum(dim=2).clamp(min=1e-6)  # [B, N]
        D_e  = H_sparse.sum(dim=1).clamp(min=1e-6)  # [B, E]
        W    = self.edge_W.abs()                     # [E]
        H_w  = H_sparse * (W / D_e).unsqueeze(1)    # [B, N, E]

        a    = torch.einsum('bne,bnd->bed', H_sparse, H)   # [B, E, D]
        out  = torch.einsum('bne,bed->bnd', H_w, a)        # [B, N, D]
        out  = out / D_v.unsqueeze(-1)                      # [B, N, D]

        return torch.tanh(out)                              # [B, N, D]  ∈ (-1,1)


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
        total_res   = cfg["total_resources"]
        k_per_edge  = cfg.get("k_per_edge", max(4, N // 32))

        # ── Stage 1 : 时间展平 + 线性投影（短/长期各一个）────────────
        self.short_proj = nn.Linear(T_short * C, D)
        self.long_proj  = nn.Linear(T_long  * C, D)

        # ── Stage 1-4 : 自适应图卷积（短/长期各一个）───────────────
        self.gcn_short = _AdaptiveGCN(N, D)
        self.gcn_long  = _AdaptiveGCN(N, D)

        # ── Stage 1-4 : 可学习节点条件嵌入 c ────────────────────────
        self.node_cond = nn.Parameter(torch.randn(N, c_dim))

        # ── Stage 2 : 弹性编码器（全局标量 τ, δ）────────────────────
        self.resil_enc = _ResilienceEncoder()

        # ── Stage 3 : Graph ODE → 稀疏超图卷积（一体化，只做一次）─
        self.hyper_conv = _GraphODEHypergraph(
            N, in_dim=D, k_per_edge=k_per_edge
        )

        # ── Stage 5-6 : RC-GLN 调制注意力层（c_aug_dim = c_dim + 2 + D）
        c_aug_dim = c_dim + 2 + D   # c ‖ log_τ(1) ‖ δ(1) ‖ x_diff(D)

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

        # ── 辅助损失投影：x_diff[D] → 1 维事件概率（供 L_diff 用）──
        self.diff_head = nn.Linear(D, 1)

        # ── 对外暴露维度（供 train.py 计算 state_dim）───────────────
        # state_hidden = h_long[D] ‖ x_diff[D] ‖ log_τ[1] ‖ δ[1]
        self.state_hidden_dim: int = 2 * D + 2

        # 用于外部读取 x_diff 计算辅助损失
        self.last_x_diff: torch.Tensor = None

    # ─────────────────────────────────────────────────────────────────────────
    def _proj_gcn(
        self,
        x:    torch.Tensor,   # [B, T, N, C]
        proj: nn.Module,      # Linear(T*C → D)
        gcn:  nn.Module,      # _AdaptiveGCN
    ) -> torch.Tensor:        # [B, N, D]
        """第一阶段：时间展平 + 线性投影 + 自适应 GCN。"""
        B, T, N, C = x.shape
        h = proj(x.permute(0, 2, 1, 3).reshape(B, N, T * C))
        return gcn(h)

    def _rc_gln(
        self,
        h:         torch.Tensor,   # [B, N, D]
        rc_layers: nn.ModuleList,  # RC-GLN 层列表
        c_aug:     torch.Tensor,   # [B, N, c_aug_dim]
    ) -> torch.Tensor:             # [B, N, D]
        """第二阶段：RC-GLN 调制注意力（以 c_aug 为条件）。"""
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

        # ── Stage 2 : 全局弹性参数 ───────────────────────────────────
        tau, delta = self.resil_enc()                 # [1], [1]  全局标量
        log_tau    = torch.log(tau + 1e-6)            # [1]

        # ── Stage 1-4 : proj + GCN（RC-GLN 之前）────────────────────
        h_long_gcn  = self._proj_gcn(long_term,  self.long_proj,  self.gcn_long)
        h_short_gcn = self._proj_gcn(short_term, self.short_proj, self.gcn_short)

        # ── Stage 3 : Graph ODE → 稀疏超图卷积（整网只做一次）──────
        # δ 作为 ODE 积分时域，控制扩散深度；扩散后特征动态构造超边
        x_diff = self.hyper_conv(h_long_gcn, delta)           # [B, N, D]
        self.last_x_diff = torch.sigmoid(
            self.diff_head(x_diff)
        )                                                      # [B, N, 1] 供外部 L_diff，不 detach 保留梯度

        # ── Stage 4-5 : c_aug = [c ‖ log_τ ‖ δ ‖ x_diff] ──────────
        # τ, δ 是全局标量，广播到 [B, N, 1]
        c = self.node_cond.unsqueeze(0).expand(B, -1, -1)          # [B, N, c_dim]
        log_tau_exp = log_tau.view(1, 1, 1).expand(B, N, 1)        # [B, N, 1]
        delta_exp   = delta.view(1, 1, 1).expand(B, N, 1)          # [B, N, 1]
        c_aug = torch.cat(
            [c, log_tau_exp, delta_exp, x_diff], dim=-1
        )  # [B, N, c_dim+3]

        # ── Stage 6 : RC-GLN 调制注意力（以 c_aug 为条件）──────────
        h_long  = self._rc_gln(h_long_gcn,  self.rc_gln_long,  c_aug)
        h_short = self._rc_gln(h_short_gcn, self.rc_gln_short, c_aug)

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
