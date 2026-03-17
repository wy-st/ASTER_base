# =============================================================================
# models/agent.py  ——  多目标 DQN 智能体（来自 ASTER）
# =============================================================================
# 包含：
#   DQN          —  Q 网络，输出 [B, N, A, 4]（4 个目标维度）
#   ReplayBuffer —  经验回放缓冲区
#   DQNAgent     —  ε-greedy 策略 + 资源约束 + double-DQN 更新
# =============================================================================

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# DQN
# ─────────────────────────────────────────────────────────────────────────────

class DQN(nn.Module):
    """
    三层 MLP，把扁平状态向量映射为多目标 Q 值。

    Q[b, n, a, o] = 第 b 个样本、第 n 个节点、动作 a 在目标 o 上的 Q 值。
    动作 a ∈ {0=不调度, 1=调度}，目标 o ∈ {成功, 误报, 距离, AET} 共 4 维。
    """

    def __init__(self, state_dim, num_nodes, num_actions=2, num_objectives=4):
        super().__init__()
        self.N = num_nodes
        self.A = num_actions
        self.O = num_objectives

        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_nodes * num_actions * num_objectives),
        )

    def forward(self, state):
        """state: [B, state_dim]  →  Q: [B, N, A, O]"""
        return self.net(state).view(state.size(0), self.N, self.A, self.O)


# ─────────────────────────────────────────────────────────────────────────────
# ReplayBuffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """固定容量的循环经验回放缓冲区。"""

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


# ─────────────────────────────────────────────────────────────────────────────
# DQNAgent
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    多目标 DQN 资源调度智能体。

    每个节点独立决策：0=不调度，1=调度。
    受两类约束：
      - 资源约束：最多调度 available_resources 个节点
      - 冷却约束：正在冷却的节点不能再次接受调度

    使用固定偏好向量 ω = [0.9, 0.05, 0.03, 0.02]
    对 4 个目标加权后做 ε-greedy 决策。

    训练使用 double-DQN（main + target 网络）。
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

        # main 网络（更新梯度） + target 网络（冻结，定期同步）
        self.main_net   = DQN(state_dim, num_nodes).to(device)
        self.target_net = DQN(state_dim, num_nodes).to(device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=cfg["lr"])
        self.buffer    = ReplayBuffer(cfg["replay_capacity"])

    # ------------------------------------------------------------------
    def sample_omega(self, B):
        """固定偏好向量，广播到批次维度。"""
        w = torch.tensor([[0.9, 0.05, 0.03, 0.02]], device=self.device)
        return w.expand(B, -1)   # [B, 4]

    # ------------------------------------------------------------------
    def select_actions(self, state, env, available_resources):
        """
        为当前样本（单步）选择每个节点的动作。

        state              : [1, state_dim] tensor
        env                : ResourceEnv，用于读取资源/冷却状态
        available_resources: int，当前可用资源数

        返回: actions [N] tensor，值在 {0, 1}
        """
        state = state.to(self.device)
        omega = self.sample_omega(1)   # [1, 4]

        with torch.no_grad():
            q   = self.main_net(state)                              # [1, N, 2, 4]
            q_s = (q * omega.view(1, 1, 1, 4)).sum(-1).squeeze(0)  # [N, 2]

        # ε-greedy 基础决策
        if np.random.rand() > self.epsilon:
            raw_act = (q_s[:, 1] > q_s[:, 0]).long()
        else:
            raw_act = torch.randint(0, 2, (self.N,), device=self.device)

        # 约束 1：冷却中的节点不能调度
        resources = torch.tensor(env.resources, device=self.device)
        cooldowns = torch.tensor(env.cooldowns, device=self.device)
        can_send  = ~((resources == 1) & (cooldowns > 0))
        raw_act[~can_send] = 0

        # 约束 2：调度总量不超过 available_resources
        if available_resources > 0 and raw_act.sum().item() > available_resources:
            q_dispatch = q_s[:, 1].clone()
            q_dispatch[~can_send] = -1e9
            topk    = torch.topk(q_dispatch, k=available_resources).indices
            raw_act = torch.zeros(self.N, dtype=torch.long, device=self.device)
            raw_act[topk] = 1

        return raw_act   # [N]

    # ------------------------------------------------------------------
    def store(self, state, action, reward_vec, next_state, done=False):
        """存入回放缓冲区（自动移到 CPU）。"""
        self.buffer.push(
            state.detach().cpu(),
            action.detach().cpu(),
            reward_vec.detach().cpu(),
            next_state.detach().cpu(),
            done,
        )

    # ------------------------------------------------------------------
    def update(self):
        """从回放缓冲区采样，做一步梯度更新。"""
        if len(self.buffer) < self.bs:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.bs)
        states      = states.to(self.device)
        actions     = actions.to(self.device)       # [B, N]
        rewards     = rewards.to(self.device)       # [B, 4]
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        omega = self.sample_omega(self.bs)          # [B, 4]

        # —— 当前 Q 值 ——
        q_all   = self.main_net(states)             # [B, N, 2, 4]
        act_idx = actions.unsqueeze(-1).unsqueeze(-1).expand(-1, self.N, 1, 4)
        q_taken = q_all.gather(2, act_idx).squeeze(2).sum(1)   # [B, 4]

        # —— 目标 Q 值（double-DQN）——
        with torch.no_grad():
            next_q    = self.target_net(next_states)            # [B, N, 2, 4]
            best_a    = next_q.mean(-1).argmax(2, keepdim=True) # [B, N, 1]
            next_take = next_q.gather(
                2, best_a.unsqueeze(-1).expand(-1, -1, -1, 4)
            ).squeeze(2).sum(1)                                  # [B, 4]
            target_q  = rewards + self.gamma * next_take * (1 - dones.unsqueeze(1))

        # —— 双重损失 ——
        # L_A：各目标维度 MSE
        loss_a = F.mse_loss(q_taken, target_q)
        # L_B：标量化后 L1（通过偏好向量加权）
        q_sc   = (omega * q_taken).sum(1)
        y_sc   = (omega * target_q).sum(1)
        loss_b = F.l1_loss(q_sc, y_sc)

        loss = 0.8 * loss_a + 0.2 * loss_b

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon 衰减 & target 网络同步
        self.steps   += 1
        self.epsilon  = max(self.eps_min, self.epsilon * self.eps_dec)
        if self.steps % self.tgt_upd == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())

        return loss.item()
