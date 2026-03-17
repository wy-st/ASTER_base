# =============================================================================
# trainer/evaluate.py  ——  验证与测试评估
# =============================================================================
# run_evaluate：在给定 DataLoader 上跑模型，计算 SR / FAR / AD / AET 指标。
#
# 指标定义：
#   SR  (Success Rate)      = 成功调度次数 / 实际事件总数
#   FAR (False Alarm Rate)  = 误报次数 / (误报 + 成功调度)
#   AD  (Average Distance)  = 平均调度距离
#   AET (Average Early Time)= 平均提前检测步数
# =============================================================================

import numpy as np
import torch

from metrics.reward import construct_rl_state, compute_reward


def run_evaluate(predictor, agent, env, data_loader, distance_matrix,
                 location, device, cfg, desc="Val"):
    """
    在 data_loader 上做完整评估，返回各项指标的均值。

    Args:
        predictor     : ResourcePredictor
        agent         : DQNAgent
        env           : ResourceEnv（会在内部 reset）
        data_loader   : DataLoader（val 或 test）
        distance_matrix: [N, N] numpy 数组
        location      : [N, 2] tensor
        device        : torch.device
        cfg           : dict
        desc          : 日志前缀（"Val" 或 "Test"）

    Returns:
        metrics : dict {sr, far, ad, aet}
    """
    predictor.eval()
    env.reset()

    total_success  = 0.0
    total_gt       = 0.0
    total_false    = 0.0
    total_dist     = 0.0
    total_aet      = 0.0
    total_samples  = 0

    with torch.no_grad():
        for short_term, long_term, target in data_loader:
            B = short_term.size(0)

            for i in range(B):
                st  = short_term[i].to(device)
                lt  = long_term[i].to(device)
                tgt = target[i].to(device)

                res_state = env.get_state().to(device)
                available = int(
                    ((res_state[:, 0] == 1) & (res_state[:, 1] == 0)).sum().item()
                )

                preds, hidden, k = predictor(
                    st.unsqueeze(0), lt.unsqueeze(0), available
                )

                rl_state = construct_rl_state(hidden, res_state, location)
                actions  = agent.select_actions(rl_state, env, available)

                k_val = int(k[0].item())
                env.k_steps = k_val
                _, total_cost = env.step(actions.cpu().tolist(), distance_matrix)

                _, comps = compute_reward(
                    tgt, actions, res_state, k_val,
                    distance_matrix, cfg, total_cost,
                )

                total_success += comps["success"]
                total_false   += comps["false_alarm"]
                total_gt      += comps["gt_event_count"]
                total_dist    += comps["distance_cost"]
                total_aet     += comps["aet"]
                total_samples += 1

    eps = 1e-8
    sr  = total_success / (total_gt    + eps)
    far = total_false   / (total_false + total_success + eps)
    ad  = total_dist    / max(1, total_samples)
    aet = total_aet     / max(1, total_samples)

    print(
        f"[{desc}]  SR={sr:.4f}  FAR={far:.4f}"
        f"  AD={ad:.4f}  AET={aet:.4f}"
    )

    predictor.train()
    return {"sr": sr, "far": far, "ad": ad, "aet": aet}
