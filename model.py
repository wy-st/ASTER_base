# =============================================================================
# model.py  —  All model code in one file
# =============================================================================
# Contents (in order):
#   1.  StandardScaler          — normalize / denormalize speed values
#   2.  TrafficDataset          — load ConFormer .npz data as ASTER-style windows
#   3.  SelfAttentionLayer      — GLN-conditioned attention (from ConFormer)
#   4.  TrafficEncoder          — spatiotemporal encoder for short/long inputs
#   5.  UnifiedResourceModule   — fuse short/long features by resource ratio
#   6.  MultiStepDecoder        — Transformer decoder for K-step prediction
#   7.  ResourcePredictor       — wraps encoders + module + decoder
#   8.  DQN                     — multi-objective Q-network
#   9.  ReplayBuffer            — experience replay storage
#   10. DQNAgent                — ε-greedy agent with resource constraint
#   11. ResourceEnv             — tracks resources & cooldowns, computes cost
#   12. RewardNormalizer        — running mean/var normalization of reward
#   13. Helper functions        — compute_reward, construct_rl_state,
#                                 make_distance_matrix, make_location_tensor
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from collections import deque
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment


# =============================================================================
# 1. StandardScaler
#    Fits mean/std on training data and normalizes speed values.
# =============================================================================

class StandardScaler:
    """Normalize and denormalize a single channel (speed)."""

    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, data):
        """data: numpy array of any shape."""
        self.mean = data.mean()
        self.std  = data.std()
        return self

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data):
        return data * self.std + self.mean


# =============================================================================
# 2. TrafficDataset
#    Reads ConFormer's .npz file and produces ASTER-style windows:
#      short_term : [T_short, N, C]   — recent context for the short encoder
#      long_term  : [T_long,  N, C]   — full context for the long encoder
#      target     : [K_max,   N, 1]   — binary event labels (speed anomaly)
# =============================================================================

class TrafficDataset(Dataset):
    """
    Converts ConFormer's raw traffic data (shape [T, N, C]) into
    (short_term, long_term, target) tuples for resource-allocation training.

    Event definition: speed at a node/time is "anomalous" (event=1) when
    the raw speed drops below  mean - threshold_sigma * std  of training speed.
    """

    def __init__(self, data, T_long, T_short, K_max, scaler, event_threshold):
        """
        Args:
            data            : np.array [T, N, C], already with speed normalized at ch 0
            T_long          : long-term window length
            T_short         : short-term window length  (≤ T_long)
            K_max           : number of future steps to predict
            scaler          : fitted StandardScaler (for inverse-transforming speed)
            event_threshold : raw-speed threshold; below this → event=1
        """
        assert T_short <= T_long, "T_short must be ≤ T_long"
        self.data            = data            # [T, N, C], speed channel is normalized
        self.T_long          = T_long
        self.T_short         = T_short
        self.K_max           = K_max
        self.scaler          = scaler
        self.event_threshold = event_threshold  # raw speed (not normalized)
        self.window          = T_long + K_max

    def __len__(self):
        return max(0, len(self.data) - self.window)

    def __getitem__(self, idx):
        # Raw window for this sample
        raw = self.data[idx : idx + self.window]          # [window, N, C]

        # ---- Inputs (normalized speed already applied) ----
        long_term  = raw[: self.T_long]                   # [T_long,  N, C]
        short_term = raw[self.T_long - self.T_short : self.T_long]  # [T_short, N, C]

        # ---- Target: future K_max steps, speed channel only ----
        future_speed_norm = raw[self.T_long :, :, 0]      # [K_max, N] — normalized speed
        # Inverse transform to get raw speed for threshold comparison
        future_speed_raw  = self.scaler.inverse_transform(future_speed_norm)
        # Binary label: 1 if speed below event threshold
        target = (future_speed_raw < self.event_threshold).astype(np.float32)
        target = target[:, :, np.newaxis]                  # [K_max, N, 1]

        return (
            torch.FloatTensor(short_term),   # [T_short, N, C]
            torch.FloatTensor(long_term),    # [T_long,  N, C]
            torch.FloatTensor(target),       # [K_max,   N, 1]
        )


def build_datasets(cfg):
    """
    Load data.npz, split 60/20/20, fit scaler on train speed,
    return (train_dataset, val_dataset, test_dataset, scaler, event_threshold).
    """
    import os
    data_path = os.path.join(cfg["data_dir"], "data.npz")
    raw = np.load(data_path)["data"].astype(np.float32)  # [T, N, F]

    T, N, F = raw.shape
    # Use only channels that exist
    C = min(F, cfg["input_dim"])
    data = raw[..., :C]                                   # [T, N, C]

    # 60 / 20 / 20 split
    train_end = int(0.6 * T)
    val_end   = int(0.8 * T)

    train_raw = data[:train_end]
    val_raw   = data[train_end:val_end]
    test_raw  = data[val_end:]

    # Fit scaler on training speed (channel 0)
    scaler = StandardScaler().fit(train_raw[..., 0])

    # Event threshold based on raw (un-normalized) training speed
    raw_train_speed = train_raw[..., 0]
    event_threshold = raw_train_speed.mean() - cfg["threshold_sigma"] * raw_train_speed.std()

    # Apply normalization to speed channel (in-place copy)
    def normalize(split):
        s = split.copy()
        s[..., 0] = scaler.transform(s[..., 0])
        return s

    train_data = normalize(train_raw)
    val_data   = normalize(val_raw)
    test_data  = normalize(test_raw)

    T_long  = cfg["T_long"]
    T_short = cfg["T_short"]
    K_max   = cfg["K_max"]

    train_ds = TrafficDataset(train_data, T_long, T_short, K_max, scaler, event_threshold)
    val_ds   = TrafficDataset(val_data,   T_long, T_short, K_max, scaler, event_threshold)
    test_ds  = TrafficDataset(test_data,  T_long, T_short, K_max, scaler, event_threshold)

    print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"[Data] Nodes: {N} | Channels: {C} | Event threshold: {event_threshold:.3f}")

    return train_ds, val_ds, test_ds, scaler, event_threshold


# =============================================================================
# 3. SelfAttentionLayer  (from ConFormer)
#    Multi-head self-attention conditioned on an external signal c via GLN.
#    GLN = Gated Linear Network: produces shift/scale/gate for both attention
#    sublayer and FFN sublayer.
#    Input:  x [B, N, model_dim],  c [B, N, c_dim]
#    Output: x [B, N, model_dim]
# =============================================================================

def _modulate(x, shift, scale):
    """Apply adaptive shift and scale: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


class SelfAttentionLayer(nn.Module):
    """
    One block of GLN-conditioned multi-head attention + FFN.
    Used inside TrafficEncoder.
    """

    def __init__(self, model_dim, c_dim, ffn_dim=256, num_heads=4, dropout=0.1):
        super().__init__()

        # Multi-head self-attention (operates across N nodes)
        self.attn     = nn.MultiheadAttention(model_dim, num_heads,
                                              dropout=dropout, batch_first=True)
        self.ln1      = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.ln2      = nn.LayerNorm(model_dim, elementwise_affine=False)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
        )
        self.drop = nn.Dropout(dropout)

        # GLN: maps conditioning c → 6 vectors (shift/scale/gate for attn & ffn)
        self.gln = nn.Sequential(
            nn.ReLU(),
            nn.Linear(c_dim, 6 * model_dim),
        )

    def forward(self, x, c):
        """
        x: [B, N, model_dim]
        c: [B, N, c_dim]   — conditioning signal (e.g. node embedding)
        """
        # Compute GLN parameters from conditioning
        params = self.gln(c)  # [B, N, 6*model_dim]
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = params.chunk(6, dim=-1)

        # --- Attention sublayer ---
        x_norm = _modulate(self.ln1(x), shift_a, scale_a)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop(gate_a * attn_out)

        # --- FFN sublayer ---
        x_norm = _modulate(self.ln2(x), shift_f, scale_f)
        x = x + self.drop(gate_f * self.ffn(x_norm))

        return x


# =============================================================================
# 4. TrafficEncoder
#    Encodes a traffic sequence [B, T, N, C] into spatial hidden states [B, N, D].
#    Step 1: Flatten time × channels and project to model_dim.
#    Step 2: Apply GLN-conditioned spatial attention (across N nodes) for L layers.
#    The conditioning c comes from a learnable per-node embedding.
# =============================================================================

class TrafficEncoder(nn.Module):
    """
    Spatiotemporal encoder inspired by ConFormer.
    - Temporal aggregation: linear projection of flattened [T, C] → model_dim
    - Spatial reasoning: L layers of GLN-conditioned multi-head attention over nodes
    """

    def __init__(self, T, input_dim, num_nodes, model_dim, c_dim,
                 num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()

        # Project (T * input_dim) → model_dim
        self.input_proj = nn.Linear(T * input_dim, model_dim)

        # Learnable per-node conditioning vector, broadcast over batch
        self.node_cond = nn.Parameter(torch.randn(num_nodes, c_dim))

        # Spatial attention layers
        self.attn_layers = nn.ModuleList([
            SelfAttentionLayer(model_dim, c_dim,
                               ffn_dim=model_dim * 4,
                               num_heads=num_heads,
                               dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: [B, T, N, C]
        Returns: [B, N, model_dim]
        """
        B, T, N, C = x.shape

        # Flatten time & channel dims: [B, N, T*C], then project → [B, N, model_dim]
        h = self.input_proj(x.permute(0, 2, 1, 3).reshape(B, N, T * C))

        # Expand node conditioning to batch: [1, N, c_dim] → [B, N, c_dim]
        c = self.node_cond.unsqueeze(0).expand(B, -1, -1)

        for layer in self.attn_layers:
            h = layer(h, c)

        return h  # [B, N, model_dim]


# =============================================================================
# 5. UnifiedResourceModule  (adapted from ASTER)
#    Fuses short-term and long-term features according to available resources.
#    More resources → rely more on long-term features (longer planning horizon).
#    Also computes k (number of future steps to predict/dispatch) from resource ratio.
# =============================================================================

class UnifiedResourceModule(nn.Module):
    """
    Combines short-term and long-term encoder outputs into a single
    fused feature, weighted by the fraction of available resources.
    Richer resources → longer prediction/dispatch horizon k.
    """

    def __init__(self, C_common, num_nodes, total_resources,
                 K_max=12, num_heads=4, dropout=0.1):
        super().__init__()
        self.C_common        = C_common
        self.num_nodes       = num_nodes
        self.total_resources = total_resources
        self.K_max           = K_max

        # Cross-node attention over fused features
        self.attn = nn.MultiheadAttention(C_common, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.Linear(C_common, C_common * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(C_common * 4, C_common),
        )
        self.ln1 = nn.LayerNorm(C_common)
        self.ln2 = nn.LayerNorm(C_common)
        self.drop = nn.Dropout(dropout)

    def forward(self, short_feat, long_feat, available_resources):
        """
        short_feat         : [B, N, C_common]
        long_feat          : [B, N, C_common]
        available_resources: scalar int or [B] tensor — number of free resources

        Returns:
            fused  : [B, N, C_common]   — fused spatial features
            k      : [B] int tensor     — predicted dispatch horizon
        """
        B, N, C = short_feat.shape

        # Resource ratio per batch item → [B, 1, 1] for broadcasting
        if isinstance(available_resources, (int, float)):
            ratio = torch.full((B,), available_resources / self.total_resources,
                               device=short_feat.device)
        else:
            available_resources = torch.as_tensor(
                available_resources, dtype=torch.float32, device=short_feat.device)
            ratio = (available_resources / self.total_resources).clamp(0.001, 1.0)

        ratio_3d = ratio.view(B, 1, 1)  # for broadcasting over N and C

        # Weighted fusion: more resources → trust long-term more
        fused = (1 - ratio_3d) * short_feat + ratio_3d * long_feat  # [B, N, C]

        # Self-attention over nodes
        attn_out, _ = self.attn(fused, fused, fused)
        fused = self.ln1(fused + self.drop(attn_out))

        # FFN
        fused = self.ln2(fused + self.drop(self.ffn(fused)))

        # Compute k: more resources → predict further into the future
        k = (ratio * self.K_max).round().clamp(min=1, max=self.K_max).long()  # [B]

        return fused, k


# =============================================================================
# 6. MultiStepDecoder  (adapted from ASTER)
#    Autoregressive Transformer decoder that generates K future predictions.
#    Each prediction is a per-node probability of an event (speed anomaly).
# =============================================================================

class MultiStepDecoder(nn.Module):
    """
    Given fused encoder memory [B, N, C], decode K future steps.
    Each step produces a per-node event probability in [0, 1].
    """

    def __init__(self, C_common, K_max=12, num_heads=4, dropout=0.1):
        super().__init__()
        self.C_common = C_common
        self.K_max    = K_max

        # Step embeddings: learnable offset for each prediction step
        self.step_embed = nn.Embedding(K_max + 1, C_common)

        # Transformer decoder layers
        dec_layer = nn.TransformerDecoderLayer(
            d_model=C_common, nhead=num_heads,
            dim_feedforward=C_common * 4, dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)

        # Output projection → event probability per node
        self.out_proj = nn.Linear(C_common, 1)

        # Learnable start token
        self.start_token = nn.Parameter(torch.randn(1, 1, C_common))

    def forward(self, memory, k):
        """
        memory : [B, N, C_common]
        k      : int — number of steps to decode

        Returns:
            predictions : [B, k, N, 1]  — event probabilities
        """
        B, N, C = memory.shape

        # Build step query embeddings: [k, C] → [B, k, C]
        step_idx   = torch.arange(k, device=memory.device)
        step_emb   = self.step_embed(step_idx)         # [k, C]
        tgt        = step_emb.unsqueeze(0).expand(B, -1, -1)  # [B, k, C]

        # Causal mask so step i only attends to steps 0..i-1
        causal_mask = torch.triu(
            torch.full((k, k), float('-inf'), device=memory.device), diagonal=1
        )

        # memory for cross-attention: [B, N, C] (treat nodes as sequence positions)
        dec_out = self.decoder(tgt, memory, tgt_mask=causal_mask)  # [B, k, C]

        # Project to per-node prediction
        # dec_out [B, k, C] → expand to [B, k, N, C] by broadcasting node embeddings
        # Simpler: project then broadcast
        out = self.out_proj(dec_out)         # [B, k, 1]
        out = out.unsqueeze(2).expand(B, k, N, 1)  # [B, k, N, 1]
        out = torch.sigmoid(out)

        return out  # [B, k, N, 1]


# =============================================================================
# 7. ResourcePredictor
#    Puts together the two encoders, UnifiedResourceModule, and MultiStepDecoder.
#    This is the main spatiotemporal prediction model.
# =============================================================================

class ResourcePredictor(nn.Module):
    """
    Full predictor model:
      short_term → short_encoder → project → ┐
                                              ├─ UnifiedResourceModule → fused
      long_term  → long_encoder  → project → ┘
                                              └─ MultiStepDecoder → predictions

    Also returns hidden states (fused features) for the DQN agent.
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

        # Short-term and long-term encoders (separate because T differs)
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

        # Project encoder outputs to shared C_common dimension
        self.proj_short = nn.Linear(D, C_common)
        self.proj_long  = nn.Linear(D, C_common)

        # Resource-aware fusion + step selector
        self.resource_module = UnifiedResourceModule(
            C_common=C_common, num_nodes=N,
            total_resources=total_res, K_max=K_max,
            num_heads=heads, dropout=dropout,
        )

        # Multi-step decoder
        self.decoder = MultiStepDecoder(
            C_common=C_common, K_max=K_max,
            num_heads=heads, dropout=dropout,
        )

    def forward(self, short_term, long_term, available_resources):
        """
        short_term         : [B, T_short, N, C]
        long_term          : [B, T_long,  N, C]
        available_resources: int or [B] tensor

        Returns:
            predictions : [B, k_max_in_batch, N, 1]  — event probabilities
            hidden      : [B, N, C_common]            — fused encoder features
            k           : [B] int tensor              — per-sample dispatch horizons
        """
        # Encode
        h_short = self.proj_short(self.short_encoder(short_term))  # [B, N, C_common]
        h_long  = self.proj_long(self.long_encoder(long_term))     # [B, N, C_common]

        # Fuse based on resource availability
        fused, k = self.resource_module(h_short, h_long, available_resources)

        # Decode for k_max steps (use the maximum k in the batch)
        k_max = int(k.max().item())
        predictions = self.decoder(fused, k_max)  # [B, k_max, N, 1]

        return predictions, fused, k


# =============================================================================
# 8. DQN  (from ASTER)
#    Multi-objective Q-network: outputs Q-values for each (node, action, objective).
#    Output shape: [B, N, num_actions, num_objectives]
# =============================================================================

class DQN(nn.Module):
    """
    3-layer MLP that maps a flat state vector to Q-values.
    Q[b, n, a, o] = Q-value for batch b, node n, action a, objective o.
    """

    def __init__(self, state_dim, num_nodes, num_actions=2, num_objectives=4):
        super().__init__()
        self.N  = num_nodes
        self.A  = num_actions
        self.O  = num_objectives

        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_nodes * num_actions * num_objectives),
        )

    def forward(self, state):
        """state: [B, state_dim] → Q: [B, N, A, O]"""
        B = state.size(0)
        return self.net(state).view(B, self.N, self.A, self.O)


# =============================================================================
# 9. ReplayBuffer  (from ASTER)
#    Stores (state, action, reward_vec, next_state, done) transitions.
# =============================================================================

class ReplayBuffer:
    """Fixed-size circular buffer for experience replay."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward_vec, next_state, done):
        self.buffer.append((state, action, reward_vec, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# 10. DQNAgent  (from ASTER)
#     Wraps DQN with ε-greedy action selection, resource constraints,
#     experience replay, and double-DQN updates.
# =============================================================================

class DQNAgent:
    """
    Multi-objective DQN agent for resource dispatch decisions.
    Each node independently decides: 0 = do nothing, 1 = dispatch resource.
    Resource and cooldown constraints are applied before returning actions.
    """

    def __init__(self, state_dim, num_nodes, cfg, device):
        self.N       = num_nodes
        self.device  = device
        self.gamma   = cfg["gamma"]
        self.epsilon = cfg["epsilon_start"]
        self.eps_dec = cfg["epsilon_decay"]
        self.eps_min = cfg["epsilon_min"]
        self.tgt_upd = cfg["target_update_freq"]
        self.bs      = cfg["rl_batch_size"]
        self.steps   = 0

        # Networks (main + frozen target)
        self.main_net   = DQN(state_dim, num_nodes).to(device)
        self.target_net = DQN(state_dim, num_nodes).to(device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=cfg["lr"])
        self.buffer    = ReplayBuffer(cfg["replay_capacity"])
        self.omega     = None  # saved preference vector (for logging)

    # ------------------------------------------------------------------
    def sample_omega(self, B):
        """Fixed preference vector over 4 objectives (success, false-alarm, dist, aet)."""
        w = torch.tensor([[0.9, 0.05, 0.03, 0.02]], device=self.device)
        return w.expand(B, -1)  # [B, 4]

    # ------------------------------------------------------------------
    def select_actions(self, state, env, available_resources):
        """
        Select dispatch actions for each node.

        state              : [1, state_dim]
        env                : ResourceEnv — used to check constraints
        available_resources: int — how many units can be dispatched now

        Returns: actions [N] in {0, 1}
        """
        state = state.to(self.device)
        omega = self.sample_omega(1)
        self.omega = omega.detach().clone()

        with torch.no_grad():
            q   = self.main_net(state)          # [1, N, 2, 4]
            q_s = (q * omega.view(1, 1, 1, 4)).sum(-1).squeeze(0)  # [N, 2]

        # ε-greedy base decision (action=1 if Q(a=1) > Q(a=0))
        if np.random.rand() > self.epsilon:
            raw_actions = (q_s[:, 1] > q_s[:, 0]).long()
        else:
            raw_actions = torch.randint(0, 2, (self.N,), device=self.device)

        # Apply resource & cooldown constraints
        resources  = torch.tensor(env.resources,  device=self.device)
        cooldowns  = torch.tensor(env.cooldowns,  device=self.device)
        can_send   = ~((resources == 1) & (cooldowns > 0))  # [N] bool
        raw_actions[~can_send] = 0

        # Cap total dispatches to available_resources
        dispatch_q   = q_s[:, 1].clone()
        dispatch_q[~can_send] = -1e9
        num_dispatch = int(raw_actions.sum().item())

        if num_dispatch > available_resources and available_resources > 0:
            topk     = torch.topk(dispatch_q, k=available_resources).indices
            final    = torch.zeros(self.N, dtype=torch.long, device=self.device)
            final[topk] = 1
            raw_actions = final

        return raw_actions  # [N]

    # ------------------------------------------------------------------
    def store(self, state, action, reward_vec, next_state, done=False):
        self.buffer.push(
            state.detach().cpu(),
            action.detach().cpu(),
            reward_vec.detach().cpu(),
            next_state.detach().cpu(),
            done,
        )

    # ------------------------------------------------------------------
    def update(self):
        """One gradient step on a sampled mini-batch."""
        if len(self.buffer) < self.bs:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.bs)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        omega = self.sample_omega(self.bs)           # [B, 4]

        # --- Q values for chosen actions ---
        q_all   = self.main_net(states)              # [B, N, 2, 4]
        act_idx = actions.unsqueeze(-1).unsqueeze(-1).expand(-1, self.N, 1, 4)
        q_taken = q_all.gather(2, act_idx).squeeze(2).sum(1)  # [B, 4]

        # --- Target Q values (double-DQN style) ---
        with torch.no_grad():
            next_q    = self.target_net(next_states)       # [B, N, 2, 4]
            best_a    = next_q.mean(-1).argmax(2, keepdim=True)  # [B, N, 1]
            next_take = next_q.gather(2, best_a.unsqueeze(-1).expand(-1, -1, -1, 4))
            next_take = next_take.squeeze(2).sum(1)       # [B, 4]
            y_vec     = rewards + self.gamma * next_take * (1 - dones.unsqueeze(1))

        # Combined loss: MSE per objective + L1 on scalarised values
        loss_a = F.mse_loss(q_taken, y_vec)
        q_sc   = (omega * q_taken).sum(1)
        y_sc   = (omega * y_vec).sum(1)
        loss_b = F.l1_loss(q_sc, y_sc)
        loss   = 0.8 * loss_a + 0.2 * loss_b

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon and sync target network
        self.steps   += 1
        self.epsilon  = max(self.eps_min, self.epsilon * self.eps_dec)
        if self.steps % self.tgt_upd == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())

        return loss.item()


# =============================================================================
# 11. ResourceEnv  (from ASTER)
#     Tracks which nodes have resources and cooldowns.
#     When a resource is dispatched to a node, that node enters cooldown
#     and cannot receive another dispatch for k_steps steps.
#     Uses the Hungarian algorithm to minimise total dispatch distance.
# =============================================================================

class ResourceEnv:
    """
    Environment tracking resource allocation across N nodes.

    resources[n] = 1 if node n currently has a resource unit, else 0.
    cooldowns[n] = remaining cooldown steps after a dispatch to node n.
    """

    def __init__(self, num_nodes, total_resources, k_steps):
        self.N               = num_nodes
        self.total_resources = total_resources
        self.k_steps         = k_steps
        self.resources       = np.zeros(num_nodes, dtype=np.int32)
        self.cooldowns       = np.zeros(num_nodes, dtype=np.int32)
        self.reset()

    def reset(self):
        """Randomly scatter resources across nodes."""
        self.resources = np.zeros(self.N, dtype=np.int32)
        self.cooldowns = np.zeros(self.N, dtype=np.int32)
        chosen = np.random.choice(self.N, self.total_resources, replace=False)
        self.resources[chosen] = 1

    def step(self, actions, distance_matrix):
        """
        Advance the environment by one time step.

        actions        : list or [N] array of {0, 1}
        distance_matrix: [N, N] pairwise distances

        Returns:
            new_available : int — number of free resources after this step
            total_cost    : float — total dispatch distance cost
        """
        # Decrement all cooldowns
        self.cooldowns = np.maximum(0, self.cooldowns - 1)

        # Use Hungarian matching to reallocate resources to requested nodes
        total_cost, (src, tgt) = self._reallocate(actions, distance_matrix)

        for s, t in zip(src, tgt):
            self.resources[s] -= 1
            self.resources[t] += 1
            self.cooldowns[t]  = self.k_steps

        new_available = int(np.sum((self.resources == 1) & (self.cooldowns == 0)))
        return new_available, total_cost

    def _reallocate(self, actions, distance_matrix):
        """Hungarian matching: move resources from free supply to demand nodes."""
        supply = np.where((self.resources == 1) & (self.cooldowns == 0))[0]
        demand = np.where(np.array(actions) == 1)[0]

        if len(supply) == 0 or len(demand) == 0:
            return 0.0, (np.array([]), np.array([]))

        cost   = distance_matrix[np.ix_(supply, demand)]
        r, c   = linear_sum_assignment(cost)
        return float(cost[r, c].sum()), (supply[r], demand[c])

    def get_state(self):
        """Return [N, 2] tensor: (resources, cooldowns) per node."""
        return torch.tensor(
            np.stack([self.resources, self.cooldowns], axis=-1),
            dtype=torch.float32,
        )


# =============================================================================
# 12. RewardNormalizer  (from ASTER)
#     Maintains running mean and variance of the 4-component reward vector
#     so the DQN target values stay in a stable range.
# =============================================================================

class RewardNormalizer:
    """Running mean/variance normalizer for a D-dimensional reward vector."""

    def __init__(self, dim=4, momentum=0.01, eps=1e-6, device="cpu"):
        self.mean     = torch.zeros(dim, device=device)
        self.var      = torch.ones(dim,  device=device)
        self.momentum = momentum
        self.eps      = eps

    def update(self, r):
        """r: [D] reward vector."""
        self.mean = (1 - self.momentum) * self.mean + self.momentum * r
        self.var  = (1 - self.momentum) * self.var  + self.momentum * (r - self.mean) ** 2

    def normalize(self, r):
        """Returns z-scored reward, L2-normalized."""
        z = (r - self.mean) / (self.var.sqrt() + self.eps)
        return z / (z.norm() + self.eps)


# =============================================================================
# 13. Helper functions
# =============================================================================

def make_distance_matrix(num_nodes):
    """
    Synthetic distance matrix for datasets without geographic coordinates.
    Nodes are laid out on a 2-D grid; distances are Euclidean.
    Units: grid cells (you can treat 1 unit ≈ 1 km).
    """
    side   = int(math.ceil(math.sqrt(num_nodes)))
    coords = np.array([(i // side, i % side) for i in range(num_nodes)],
                      dtype=np.float32)
    diff   = coords[:, None, :] - coords[None, :, :]  # [N, N, 2]
    dist   = np.sqrt((diff ** 2).sum(-1))              # [N, N]
    return dist


def make_location_tensor(num_nodes, device):
    """
    Synthetic (x, y) location for each node, normalized to [0, 1].
    Used as part of the DQN state so the agent can reason about geography.
    """
    side   = int(math.ceil(math.sqrt(num_nodes)))
    coords = np.array([(i // side, i % side) for i in range(num_nodes)],
                      dtype=np.float32)
    coords = coords / side                             # [N, 2] in [0, 1]
    return torch.FloatTensor(coords).to(device)


def construct_rl_state(hidden, res_state, location):
    """
    Build the flat DQN input vector from three per-node feature sets.

    hidden    : [B, N, C_common]   — spatiotemporal encoder output
    res_state : [N, 2]             — resources and cooldowns (from env.get_state())
    location  : [N, 2]             — normalized grid coordinates

    Returns:  [B, N*(C_common + 4)]
    """
    B, N, C = hidden.shape

    # Expand res_state and location to batch dimension
    res = res_state.unsqueeze(0).expand(B, -1, -1)   # [B, N, 2]
    loc = location.unsqueeze(0).expand(B, -1, -1)    # [B, N, 2]

    combined = torch.cat([hidden, res, loc], dim=-1)  # [B, N, C+4]
    return combined.view(B, -1)                        # [B, N*(C+4)]


def compute_reward(target, action, prev_res_state, k, dist_matrix, cfg, total_cost):
    """
    Compute scalar reward and its 4 components for a single sample.

    target        : [K, N, 1] float — binary event labels (already in {0, 1})
    action        : [N]       long  — dispatch decisions
    prev_res_state: [N, 2]   float  — (resources, cooldowns) before this step
    k             : int             — dispatch horizon used this step
    dist_matrix   : [N, N]   float  — pairwise distances
    cfg           : dict            — reward weight config
    total_cost    : float           — reallocation cost from env.step()

    Returns: (scalar_reward, components_dict)
    """
    target  = target[:k]                          # [k, N, 1]
    t_bin   = (target > 0.5).float()              # binary events
    gt_agg  = t_bin.max(dim=0).values             # [N, 1] — any event in [0,k)?
    gt_count = (gt_agg > 0.5).sum().item()

    act      = action.float()                     # [N]
    has_res  = (prev_res_state[:, 0] == 1)        # [N] bool
    cooldown = prev_res_state[:, 1]               # [N]

    # True positives (dispatched & event occurred)
    tp  = (act > 0) & (gt_agg.squeeze(-1) > 0)
    # Already-dispatched resources that cover the event
    prev_ok = (act == 0) & has_res & (cooldown >= k) & (gt_agg.squeeze(-1) > 0)
    success  = (tp.sum() + prev_ok.sum()).item()

    # False positives (dispatched but no event)
    fp       = (act > 0) & (gt_agg.squeeze(-1) == 0)
    false_alm = fp.sum().item()

    # Average earliest event time at true-positive nodes
    aet_vals  = []
    for t in range(k):
        hit = t_bin[t].squeeze(-1) > 0
        for n_idx in torch.where(tp)[0]:
            if hit[n_idx] and n_idx.item() not in [x[0] for x in aet_vals]:
                aet_vals.append((n_idx.item(), t))
    aet = np.mean([v[1] for v in aet_vals]) if aet_vals else 0.0

    # Scalar reward
    r = (
        cfg["reward_alpha"]  * success
      - cfg["reward_beta"]   * false_alm
      - cfg["reward_gamma"]  * total_cost
      + cfg["reward_delta"]  * aet
    )

    components = {
        "success":       success,
        "false_alarm":   false_alm,
        "distance_cost": total_cost,
        "aet":           aet,
        "total_reward":  r,
        "gt_event_count": gt_count,
    }
    return r, components


def compute_predictor_loss(predictions, target, k):
    """
    Masked MSE loss between predicted and true event probabilities.

    predictions : [B, K_max, N, 1]
    target      : [B, K_max, N, 1]
    k           : [B] int tensor — only compute loss for first k steps

    Returns: scalar loss
    """
    B, K, N, _ = predictions.shape
    step_range  = torch.arange(K, device=predictions.device).unsqueeze(0)   # [1, K]
    mask        = (step_range < k.unsqueeze(1)).unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
    mask        = mask.expand_as(predictions)

    loss        = F.mse_loss(predictions, target[..., :1], reduction="none")
    masked      = (loss * mask).sum()
    count       = mask.sum().clamp(min=1)
    return masked / count
