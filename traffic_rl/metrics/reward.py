# =============================================================================
# metrics/reward.py  ——  奖励计算与预测损失
# =============================================================================
# 包含：
#   compute_reward          —  计算单步标量奖励及其 4 个分量
#   compute_predictor_loss  —  带掩码的 MSE 预测损失（只算前 k 步）
#   construct_rl_state      —  拼接 DQN 状态向量
# =============================================================================

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# construct_rl_state
# ─────────────────────────────────────────────────────────────────────────────

def construct_rl_state(hidden, res_state, location):
    """
    把三组每节点特征拼接成 DQN 输入的扁平状态向量。

    Args:
        hidden    : [B, N, C_common]  —— 预测模型输出的隐藏特征
        res_state : [N, 2]            —— 资源状态（resources, cooldowns）
        location  : [N, 2]            —— 节点位置（归一化坐标）

    Returns:
        state : [B, N * (C_common + 4)]
    """
    B, N, C = hidden.shape
    res = res_state.unsqueeze(0).expand(B, -1, -1)   # [B, N, 2]
    loc = location.unsqueeze(0).expand(B, -1, -1)    # [B, N, 2]

    combined = torch.cat([hidden, res, loc], dim=-1)  # [B, N, C+4]
    return combined.view(B, -1)                        # [B, N*(C+4)]


# ─────────────────────────────────────────────────────────────────────────────
# compute_reward
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(target, action, prev_res_state, k, dist_matrix, cfg, total_cost):
    """
    计算单个样本的调度奖励及其分量。

    奖励由 4 个分量线性组合：
        r = α·success - β·false_alarm - γ·distance_cost + δ·aet

    Args:
        target        : [K, N, 1] float  —— 二值事件标签（0/1）
        action        : [N]       long   —— 调度决策（0/1）
        prev_res_state: [N, 2]    float  —— 调度前的（资源, 冷却）状态
        k             : int              —— 本次使用的预测步数
        dist_matrix   : [N, N]   float  —— 节点间距离
        cfg           : dict             —— 含 reward_alpha/beta/gamma/delta
        total_cost    : float            —— 调度距离（来自 ResourceEnv.step）

    Returns:
        reward     : float
        components : dict {success, false_alarm, distance_cost, aet,
                           total_reward, gt_event_count}
    """
    target  = target[:k]                              # [k, N, 1]
    t_bin   = (target > 0.5).float()                  # 二值化
    gt_agg  = t_bin.max(dim=0).values.squeeze(-1)     # [N] 在 k 步内是否有事件
    gt_count = (gt_agg > 0.5).sum().item()

    act      = action.float()                         # [N]
    has_res  = (prev_res_state[:, 0] == 1)            # [N] 是否持有资源
    cooldown = prev_res_state[:, 1]                   # [N] 冷却时间

    # —— 成功调度（True Positive）——
    # 1. 当前决策调度，且该节点未来有事件
    tp       = (act > 0) & (gt_agg > 0)
    # 2. 已在该节点驻守（resources=1, cooldowns≥k），且有事件
    prev_ok  = (act == 0) & has_res & (cooldown >= k) & (gt_agg > 0)
    success  = (tp.sum() + prev_ok.sum()).item()

    # —— 误报（False Positive）——
    fp        = (act > 0) & (gt_agg == 0)
    false_alm = fp.sum().item()

    # —— 平均最早事件时刻（AET：越早发现越好）——
    aet_list = []
    for t_idx in range(k):
        hit_nodes = t_bin[t_idx].squeeze(-1) > 0
        for n_idx in torch.where(tp)[0]:
            # 记录节点第一次触发事件的时间步
            if hit_nodes[n_idx] and not any(n_idx.item() == e[0] for e in aet_list):
                aet_list.append((n_idx.item(), t_idx))
    aet = np.mean([e[1] for e in aet_list]) if aet_list else 0.0

    # —— 标量奖励 ——
    r = (
        cfg["reward_alpha"]  * success
      - cfg["reward_beta"]   * false_alm
      - cfg["reward_gamma"]  * total_cost
      + cfg["reward_delta"]  * aet
    )

    components = {
        "success":        success,
        "false_alarm":    false_alm,
        "distance_cost":  total_cost,
        "aet":            aet,
        "total_reward":   r,
        "gt_event_count": gt_count,
    }
    return r, components


# ─────────────────────────────────────────────────────────────────────────────
# compute_predictor_loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_predictor_loss(predictions, target, k):
    """
    带掩码的 MSE 损失：只计算每个样本前 k[b] 步的误差。

    Args:
        predictions : [B, K_max, N, 1]  —— 模型输出的事件概率
        target      : [B, K_max, N, 1]  —— 二值标签
        k           : [B] int tensor    —— 每样本有效步数

    Returns:
        loss : scalar tensor
    """
    B, K, N, _ = predictions.shape

    # mask[b, t] = True 当 t < k[b]
    step_range = torch.arange(K, device=predictions.device).unsqueeze(0)   # [1, K]
    mask       = (step_range < k.unsqueeze(1))                              # [B, K]
    mask       = mask.unsqueeze(-1).unsqueeze(-1).expand_as(predictions)   # [B, K, N, 1]

    element_loss = F.mse_loss(predictions, target[..., :1], reduction="none")
    masked_loss  = (element_loss * mask).sum()
    count        = mask.sum().clamp(min=1)

    return masked_loss / count
