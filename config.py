# =============================================================================
# config.py  —  All hyperparameters in one place
# =============================================================================
# How to use:
#   1. Set ACTIVE_DATASET to "METRLA", "PEMSBAY", or "TKY"
#   2. Adjust data_dir to point to your data.npz file
#   3. Run:  python train.py
# =============================================================================

# ---- Change this to switch datasets ----------------------------------------
ACTIVE_DATASET = "METRLA"

# ---- Dataset-specific configs -----------------------------------------------
CONFIGS = {

    # ------------------------------------------------------------------
    # METR-LA  (207 road sensors in Los Angeles)
    # data.npz shape: [T, 207, 3]  (speed, time-of-day, day-of-week)
    # ------------------------------------------------------------------
    "METRLA": {
        # --- Data ---
        "data_dir":   "/home/user/ConFormer_-base/data/METRLA",
        "num_nodes":  207,
        "input_dim":  3,        # channels: speed + ToD + DoW
        "steps_per_day": 288,   # 5-min intervals → 288 steps/day

        # --- Sliding window sizes ---
        "T_long":  36,          # long-term context window  (36 steps = 3 hours)
        "T_short": 12,          # short-term context window (12 steps = 1 hour)
        "K_max":   12,          # max prediction / dispatch horizon (1 hour)

        # --- Event detection ---
        # A traffic "event" is detected when speed drops below:
        #   event_threshold = mean - threshold_sigma * std   (on raw speed)
        "threshold_sigma": 0.5,

        # --- Model architecture ---
        "model_dim":  64,       # hidden dimension inside encoder
        "c_dim":      64,       # conditioning vector dim (= model_dim here)
        "C_common":   32,       # shared hidden dim after projection
        "num_heads":  4,        # attention heads
        "num_layers": 3,        # encoder attention layers
        "dropout":    0.1,

        # --- Resource allocation ---
        "total_resources":  30, # total dispatch units (e.g. ambulances / repair crews)
        "num_actions":       2, # per-node: {0 = do nothing, 1 = dispatch}

        # --- DQN / RL ---
        "gamma":              0.99,
        "epsilon_start":      1.0,
        "epsilon_decay":      0.995,
        "epsilon_min":        0.05,
        "replay_capacity":    5000,
        "rl_batch_size":      32,
        "target_update_freq": 100,

        # --- Reward weights (see compute_reward in model.py) ---
        "reward_alpha": 1.0,    # weight for successful dispatches
        "reward_beta":  0.01,   # penalty for false alarms
        "reward_gamma": 0.01,   # penalty for dispatch distance
        "reward_delta": 0.3,    # reward for early detection

        # --- Training ---
        "batch_size":  16,
        "num_epochs":  50,
        "lr":          0.001,
        "early_stop_patience": 10,  # stop if val SR doesn't improve for N epochs

        # --- Misc ---
        "seed":   42,
        "device": "cuda",       # "cuda" or "cpu"
    },

    # ------------------------------------------------------------------
    # PEMS-BAY  (325 sensors in the Bay Area)
    # data.npz shape: [T, 325, 3]
    # ------------------------------------------------------------------
    "PEMSBAY": {
        "data_dir":   "/home/user/ConFormer_-base/data/PEMSBAY",
        "num_nodes":  325,
        "input_dim":  3,
        "steps_per_day": 288,

        "T_long":  36,
        "T_short": 12,
        "K_max":   12,

        "threshold_sigma": 0.5,

        "model_dim":  64,
        "c_dim":      64,
        "C_common":   32,
        "num_heads":  4,
        "num_layers": 3,
        "dropout":    0.1,

        "total_resources":  40,
        "num_actions":       2,

        "gamma":              0.99,
        "epsilon_start":      1.0,
        "epsilon_decay":      0.995,
        "epsilon_min":        0.05,
        "replay_capacity":    5000,
        "rl_batch_size":      32,
        "target_update_freq": 100,

        "reward_alpha": 1.0,
        "reward_beta":  0.01,
        "reward_gamma": 0.01,
        "reward_delta": 0.3,

        "batch_size":  16,
        "num_epochs":  50,
        "lr":          0.001,
        "early_stop_patience": 10,

        "seed":   42,
        "device": "cuda",
    },

    # ------------------------------------------------------------------
    # TKY  (1843 road segments in Tokyo)
    # data.npz shape: [T, 1843, 1]  — speed only, no ToD/DoW
    # ------------------------------------------------------------------
    "TKY": {
        "data_dir":   "/home/user/ConFormer_-base/data/TKY",
        "num_nodes":  1843,
        "input_dim":  1,        # only speed channel
        "steps_per_day": 144,   # 10-min intervals → 144 steps/day

        "T_long":  24,          # long-term window  (24 steps = 4 hours)
        "T_short":  6,          # short-term window  (6 steps = 1 hour)
        "K_max":    6,          # max prediction / dispatch horizon

        "threshold_sigma": 0.5,

        "model_dim":  64,
        "c_dim":      64,
        "C_common":   32,
        "num_heads":  2,        # fewer heads because TKY is larger
        "num_layers": 3,
        "dropout":    0.1,

        "total_resources":  100,
        "num_actions":        2,

        "gamma":              0.99,
        "epsilon_start":      1.0,
        "epsilon_decay":      0.995,
        "epsilon_min":        0.05,
        "replay_capacity":    5000,
        "rl_batch_size":      32,
        "target_update_freq": 100,

        "reward_alpha": 1.0,
        "reward_beta":  0.01,
        "reward_gamma": 0.01,
        "reward_delta": 0.3,

        "batch_size":  4,       # smaller batch because 1843 nodes is large
        "num_epochs":  50,
        "lr":          0.001,
        "early_stop_patience": 10,

        "seed":   42,
        "device": "cuda",
    },
}

# Active config — this is what train.py imports
CFG = CONFIGS[ACTIVE_DATASET]
