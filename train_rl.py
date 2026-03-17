# =============================================================================
# train.py  —  Single entry point.  Just run:  python train.py
# =============================================================================
# Steps performed automatically:
#   1. Load config from config.py
#   2. Load traffic data (ConFormer .npz) and build ASTER-style datasets
#   3. Initialize ResourcePredictor, DQNAgent, ResourceEnv, RewardNormalizer
#   4. For each epoch:
#        - For each batch sample: run RL dispatch loop, compute rewards, update
#   5. Validate on val set, track Success Rate; early stop if no improvement
# =============================================================================

import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# ---------- Import everything from model.py and config.py --------------------
from config import CFG
from model import (
    build_datasets,
    ResourcePredictor,
    DQNAgent,
    ResourceEnv,
    RewardNormalizer,
    construct_rl_state,
    compute_reward,
    compute_predictor_loss,
    make_distance_matrix,
    make_location_tensor,
)


# =============================================================================
# Reproducibility seed
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# Validation metric: Success Rate (SR) over the validation set
# =============================================================================

def evaluate(predictor, agent, env, val_loader, distance_matrix, location, device, cfg):
    """
    Run the model on the validation set and compute:
      SR  = successful dispatches / total ground-truth events
      FAR = false alarms / (false alarms + successful dispatches)

    Returns: (SR, FAR) as floats.
    """
    predictor.eval()
    total_success  = 0
    total_gt       = 0
    total_false    = 0

    with torch.no_grad():
        for short_term, long_term, target in val_loader:
            B = short_term.size(0)

            for i in range(B):
                st = short_term[i].to(device)   # [T_short, N, C]
                lt = long_term[i].to(device)     # [T_long,  N, C]
                tgt = target[i].to(device)        # [K_max,   N, 1]

                res_state = env.get_state().to(device)
                available = int(((res_state[:, 0] == 1) & (res_state[:, 1] == 0)).sum().item())

                preds, hidden, k = predictor(
                    st.unsqueeze(0), lt.unsqueeze(0), available
                )

                rl_state = construct_rl_state(hidden, res_state, location)
                actions  = agent.select_actions(rl_state, env, available)

                k_val = int(k[0].item())
                tgt_k = tgt[:k_val]                             # [k, N, 1]
                t_bin = (tgt_k > 0.5).float()
                gt_agg = t_bin.max(dim=0).values.squeeze(-1)   # [N]

                tp = ((actions == 1).float() * (gt_agg > 0).float()).sum().item()
                fp = ((actions == 1).float() * (gt_agg == 0).float()).sum().item()
                gt = (gt_agg > 0).sum().item()

                total_success += tp
                total_false   += fp
                total_gt      += gt

                env.step(actions.cpu().tolist(), distance_matrix)

    sr  = total_success / (total_gt + 1e-8)
    far = total_false   / (total_false + total_success + 1e-8)
    predictor.train()
    return sr, far


# =============================================================================
# Train one epoch: iterate over all samples in the DataLoader
# =============================================================================

def train_one_epoch(
    predictor, agent, env, train_loader,
    optimizer, reward_normalizer, distance_matrix, location,
    device, cfg, epoch,
):
    """
    Runs one full pass over the training data.
    Returns average reward and average predictor loss for this epoch.
    """
    predictor.train()
    total_reward = 0.0
    total_pred_loss = 0.0
    total_samples   = 0

    for batch_idx, (short_term, long_term, target) in enumerate(train_loader):
        B = short_term.size(0)

        for i in range(B):
            # ---- Unpack one sample ----------------------------------------
            st  = short_term[i].to(device)   # [T_short, N, C]
            lt  = long_term[i].to(device)     # [T_long,  N, C]
            tgt = target[i].to(device)         # [K_max,   N, 1]

            # ---- Current resource state ------------------------------------
            res_state = env.get_state().to(device)  # [N, 2]
            available = int(
                ((res_state[:, 0] == 1) & (res_state[:, 1] == 0)).sum().item()
            )

            # ---- Predictor forward pass ------------------------------------
            preds, hidden, k = predictor(
                st.unsqueeze(0),   # [1, T_short, N, C]
                lt.unsqueeze(0),   # [1, T_long,  N, C]
                available,
            )
            # preds: [1, k_max, N, 1], hidden: [1, N, C_common], k: [1]

            # ---- Build RL state and select actions -------------------------
            rl_state = construct_rl_state(hidden, res_state, location)  # [1, state_dim]
            actions  = agent.select_actions(rl_state, env, available)    # [N]

            # ---- Advance environment ----------------------------------------
            env.k_steps = int(k[0].item())
            new_avail, total_cost = env.step(
                actions.cpu().tolist(), distance_matrix
            )

            # ---- Compute next state (for replay buffer) --------------------
            # Slide window: drop earliest k steps, append k target steps
            k_val  = int(k[0].item())
            new_st_data = torch.cat([st[k_val:], tgt[:k_val]], dim=0)    # [T_short, N, C] approx
            new_lt_data = torch.cat([lt[k_val:], tgt[:k_val]], dim=0)    # [T_long,  N, C] approx

            # Pad or truncate to original window lengths
            T_short = st.shape[0]
            T_long  = lt.shape[0]
            new_st_data = _pad_or_trim(new_st_data, T_short)
            new_lt_data = _pad_or_trim(new_lt_data, T_long)

            next_avail_t = torch.tensor([new_avail], dtype=torch.float32, device=device)
            with torch.no_grad():
                _, next_hidden, _ = predictor(
                    new_st_data.unsqueeze(0),
                    new_lt_data.unsqueeze(0),
                    new_avail,
                )
            next_res  = env.get_state().to(device)
            next_state = construct_rl_state(next_hidden, next_res, location)

            # ---- Compute reward ---------------------------------------------
            reward, comps = compute_reward(
                tgt, actions, res_state, k_val,
                distance_matrix, cfg, total_cost
            )

            reward_vec = torch.tensor(
                [
                    10.0  * comps["success"],
                    -0.5  * comps["false_alarm"],
                    -0.1  * comps["distance_cost"],
                     1.0  * comps["aet"],
                ],
                dtype=torch.float32, device=device,
            )
            reward_normalizer.update(reward_vec.detach())
            norm_reward = reward_normalizer.normalize(reward_vec)

            # ---- Store transition & update DQN -----------------------------
            agent.store(
                rl_state.squeeze(0),
                actions,
                norm_reward,
                next_state.squeeze(0),
            )
            rl_loss = agent.update()

            # ---- Predictor supervised loss ---------------------------------
            pred_loss = compute_predictor_loss(
                preds,                          # [1, k_max, N, 1]
                tgt.unsqueeze(0),               # [1, K_max, N, 1]
                k,                              # [1]
            )
            optimizer.zero_grad()
            pred_loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=5.0)
            optimizer.step()

            total_reward   += reward
            total_pred_loss += pred_loss.item()
            total_samples   += 1

        # ---- Periodic progress print ---------------------------------------
        if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
            avg_r = total_reward   / max(1, total_samples)
            avg_l = total_pred_loss / max(1, total_samples)
            print(f"  [Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)}]"
                  f"  avg_reward={avg_r:.4f}  pred_loss={avg_l:.4f}"
                  f"  ε={agent.epsilon:.3f}")

    avg_reward   = total_reward   / max(1, total_samples)
    avg_pred_loss = total_pred_loss / max(1, total_samples)
    return avg_reward, avg_pred_loss


def _pad_or_trim(tensor, target_len):
    """Ensure the first dimension of tensor equals target_len (pad zeros or trim)."""
    cur = tensor.shape[0]
    if cur == target_len:
        return tensor
    elif cur < target_len:
        pad = torch.zeros(target_len - cur, *tensor.shape[1:],
                          device=tensor.device, dtype=tensor.dtype)
        return torch.cat([pad, tensor], dim=0)
    else:
        return tensor[-target_len:]


# =============================================================================
# Main training loop
# =============================================================================

def main():
    cfg    = CFG
    seed   = cfg.get("seed", 42)
    set_seed(seed)

    # ---- Device -------------------------------------------------------------
    device_str = cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"[Config] Dataset={cfg['data_dir'].split('/')[-1]}  Device={device}")

    # ---- Data ---------------------------------------------------------------
    print("[Data] Loading and splitting ...")
    train_ds, val_ds, test_ds, scaler, event_thr = build_datasets(cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"],
                              shuffle=False, drop_last=False)

    # ---- Distance matrix & location tensor ----------------------------------
    N = cfg["num_nodes"]
    distance_matrix = make_distance_matrix(N)            # [N, N] numpy
    location        = make_location_tensor(N, device)    # [N, 2] tensor

    # ---- Models -------------------------------------------------------------
    print("[Model] Initializing ...")
    predictor = ResourcePredictor(cfg).to(device)

    C_common   = cfg["C_common"]
    state_dim  = N * (C_common + 4)   # hidden(C_common) + resources(1) + cooldown(1) + xy(2)
    agent      = DQNAgent(state_dim, N, cfg, device)

    env        = ResourceEnv(N, cfg["total_resources"], cfg["K_max"])
    normalizer = RewardNormalizer(dim=4, device=device)

    optimizer  = optim.Adam(predictor.parameters(), lr=cfg["lr"])

    # Count trainable parameters
    n_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"[Model] Predictor trainable params: {n_params:,}")

    # ---- Training loop ------------------------------------------------------
    best_sr      = 0.0
    patience_cnt = 0
    patience     = cfg.get("early_stop_patience", 10)

    print(f"\n[Train] Starting training for {cfg['num_epochs']} epochs ...")
    print(f"[Train] Early stopping patience = {patience} epochs\n")

    for epoch in range(cfg["num_epochs"]):
        env.reset()   # reset resources at the start of each epoch

        avg_reward, avg_loss = train_one_epoch(
            predictor, agent, env, train_loader,
            optimizer, normalizer, distance_matrix, location,
            device, cfg, epoch,
        )

        # Evaluate on val set
        env.reset()
        sr, far = evaluate(predictor, agent, env, val_loader,
                           distance_matrix, location, device, cfg)

        print(f"[Epoch {epoch+1:3d}/{cfg['num_epochs']}]"
              f"  reward={avg_reward:.4f}"
              f"  pred_loss={avg_loss:.4f}"
              f"  val_SR={sr:.4f}"
              f"  val_FAR={far:.4f}"
              f"  ε={agent.epsilon:.3f}"
              + (" ← best" if sr > best_sr else ""))

        # Early stopping
        if sr > best_sr:
            best_sr      = sr
            patience_cnt = 0
            # Save best model weights
            torch.save(predictor.state_dict(), "best_predictor.pt")
            torch.save(agent.main_net.state_dict(), "best_dqn.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n[Early Stop] No SR improvement for {patience} epochs. Stopping.")
                break

    # ---- Final test evaluation ----------------------------------------------
    print("\n[Test] Loading best model and evaluating on test set ...")
    predictor.load_state_dict(torch.load("best_predictor.pt", map_location=device))
    agent.main_net.load_state_dict(torch.load("best_dqn.pt", map_location=device))
    env.reset()
    test_sr, test_far = evaluate(predictor, agent, env, test_loader,
                                 distance_matrix, location, device, cfg)

    print(f"\n{'='*50}")
    print(f"  Final Test Results")
    print(f"  Success Rate (SR)  : {test_sr:.4f}")
    print(f"  False Alarm Rate   : {test_far:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
