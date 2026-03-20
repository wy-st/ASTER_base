# =============================================================================
# trainer/train_epoch.py  ——  单个 epoch 的训练逻辑
# =============================================================================
# run_one_epoch：
#   遍历 DataLoader 中的所有批次，
#   对每个样本依次执行：
#     1. 预测模型 forward
#     2. RL 状态构建 + 动作选择
#     3. 环境推进 + 奖励计算
#     4. 存入回放缓冲区 + DQN 更新
#     5. 预测损失反向传播
# =============================================================================

import numpy as np
import torch
import torch.nn.functional as F

from metrics.reward import construct_rl_state, compute_reward, compute_predictor_loss


# ─────────────────────────────────────────────────────────────────────────────
# 工具：把滑动窗口向前推进 k 步
# ─────────────────────────────────────────────────────────────────────────────

def _slide_window(old_seq, new_steps, target_len):
    """
    用 new_steps 追加到 old_seq 末尾，再截取最后 target_len 步。

    old_seq  : [T, N, C]
    new_steps: [k, N, C]  （或 [k, N, 1]，会自动对齐通道数）
    返回     : [target_len, N, C]
    """
    # 对齐通道数（target 只有 1 个通道时，用零填充其余通道）
    T_old, N, C = old_seq.shape
    k, _, C_new = new_steps.shape
    if C_new < C:
        pad = torch.zeros(k, N, C - C_new, device=old_seq.device)
        new_steps = torch.cat([new_steps, pad], dim=-1)
    elif C_new > C:
        new_steps = new_steps[..., :C]

    merged = torch.cat([old_seq, new_steps], dim=0)  # [T+k, N, C]
    return merged[-target_len:]                        # [target_len, N, C]


# ─────────────────────────────────────────────────────────────────────────────
# run_one_epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_one_epoch(
    predictor,
    agent,
    env,
    train_loader,
    optimizer,
    reward_normalizer,
    distance_matrix,
    location,
    device,
    cfg,
    epoch,
):
    """
    遍历一个完整 epoch，逐样本做 RL + 有监督训练。

    Returns:
        avg_reward    : float，该 epoch 平均奖励
        avg_pred_loss : float，该 epoch 平均预测损失
        avg_components: dict，各奖励分量的平均值
    """
    predictor.train()

    total_reward     = 0.0
    total_pred_loss  = 0.0
    total_samples    = 0
    component_sums   = {
        "success": 0.0, "false_alarm": 0.0,
        "distance_cost": 0.0, "aet": 0.0,
    }

    n_batches = len(train_loader)
    log_every = max(1, n_batches // 5)   # 每 epoch 打印 5 次进度

    for batch_idx, (short_term, long_term, target) in enumerate(train_loader):
        B = short_term.size(0)

        for i in range(B):
            # ── 取出单个样本 ────────────────────────────────────────────
            st  = short_term[i].to(device)   # [T_short, N, C]
            lt  = long_term[i].to(device)    # [T_long,  N, C]
            tgt = target[i].to(device)        # [K_max,   N, 1]

            # ── 当前资源状态 ────────────────────────────────────────────
            res_state = env.get_state().to(device)   # [N, 2]
            available = int(
                ((res_state[:, 0] == 1) & (res_state[:, 1] == 0)).sum().item()
            )

            # ── 预测模型 forward ────────────────────────────────────────
            preds, hidden, k = predictor(
                st.unsqueeze(0),    # [1, T_short, N, C]
                lt.unsqueeze(0),    # [1, T_long,  N, C]
                available,
            )
            # preds: [1, k_max, N, 1], hidden: [1, N, C_common], k: [1]

            # ── 构建 RL 状态，选择动作 ──────────────────────────────────
            rl_state = construct_rl_state(hidden, res_state, location)  # [1, state_dim]
            actions  = agent.select_actions(rl_state, env, available)   # [N]

            # ── 环境推进 ────────────────────────────────────────────────
            k_val = int(k[0].item())
            env.k_steps = k_val
            new_avail, total_cost = env.step(
                actions.cpu().tolist(), distance_matrix
            )

            # ── 计算下一状态（用于存入回放缓冲区）──────────────────────
            new_st = _slide_window(st, tgt[:k_val], st.shape[0])   # [T_short, N, C]
            new_lt = _slide_window(lt, tgt[:k_val], lt.shape[0])   # [T_long,  N, C]

            with torch.no_grad():
                _, next_hidden, _ = predictor(
                    new_st.unsqueeze(0),
                    new_lt.unsqueeze(0),
                    new_avail,
                )
            next_res_state = env.get_state().to(device)
            next_rl_state  = construct_rl_state(next_hidden, next_res_state, location)

            # ── 计算奖励 ────────────────────────────────────────────────
            reward, comps = compute_reward(
                tgt, actions, res_state, k_val,
                distance_matrix, cfg, total_cost,
            )

            reward_vec = torch.tensor(
                [
                    10.0 * comps["success"],
                    -0.5 * comps["false_alarm"],
                    -0.1 * comps["distance_cost"],
                     1.0 * comps["aet"],
                ],
                dtype=torch.float32, device=device,
            )
            reward_normalizer.update(reward_vec.detach())
            norm_reward = reward_normalizer.normalize(reward_vec)

            # ── 存入经验回放 + 更新 DQN ────────────────────────────────
            agent.store(
                rl_state.squeeze(0),
                actions,
                norm_reward,
                next_rl_state.squeeze(0),
            )
            agent.update()

            # ── 预测模型有监督损失 ──────────────────────────────────────
            pred_loss = compute_predictor_loss(preds, tgt.unsqueeze(0), k)

            # ── L_diff 辅助损失（仅 Hyperion 使用，其他模型无 last_x_diff）──
            # tgt 已是二值标签 [K_max, N, 1]，取前 k_val 步内任一步发生事件为正样本
            x_diff_pred = getattr(predictor, 'last_x_diff', None)
            if x_diff_pred is not None:
                event_tgt = (
                    tgt[:k_val, :, 0:1]          # [k_val, N, 1]
                    .max(dim=0).values            # [N, 1]  k步内有事件=1
                    .unsqueeze(0)                 # [1, N, 1] → 广播到 [B, N, 1]
                    .expand_as(x_diff_pred)
                    .to(device)
                )
                l_diff    = F.binary_cross_entropy(x_diff_pred, event_tgt)
                pred_loss = pred_loss + 0.1 * l_diff

            optimizer.zero_grad()
            pred_loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=5.0)
            optimizer.step()

            # ── 累积统计 ────────────────────────────────────────────────
            total_reward    += reward
            total_pred_loss += pred_loss.item()
            total_samples   += 1
            for key in component_sums:
                component_sums[key] += comps[key]

        # ── 打印进度 ────────────────────────────────────────────────────
        if (batch_idx + 1) % log_every == 0:
            avg_r = total_reward    / max(1, total_samples)
            avg_l = total_pred_loss / max(1, total_samples)
            print(
                f"  [Epoch {epoch+1} | Batch {batch_idx+1}/{n_batches}]"
                f"  reward={avg_r:.4f}  pred_loss={avg_l:.4f}"
                f"  ε={agent.epsilon:.3f}"
            )

    denom = max(1, total_samples)
    avg_components = {k: v / denom for k, v in component_sums.items()}

    return (
        total_reward    / denom,
        total_pred_loss / denom,
        avg_components,
    )
