# =============================================================================
# train.py  ——  唯一入口，直接运行：python train.py
# =============================================================================
# 步骤：
#   1. 读取 config.py 中的 CFG
#   2. 加载数据，构建 DataLoader
#   3. 初始化 ResourcePredictor / DQNAgent / ResourceEnv / RewardNormalizer
#   4. 逐 epoch 训练，验证 SR/FAR，早停并保存最优模型
#   5. 测试集最终评估
# =============================================================================

import os
import sys
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 把项目根目录加到路径，使子包可以直接 import
sys.path.insert(0, os.path.dirname(__file__))

from config import CFG
from data.dataset import build_datasets
from models.predictor import ResourcePredictor
from models.agent import DQNAgent
from utils.env import ResourceEnv
from utils.normalizer import RewardNormalizer
from utils.graph import make_distance_matrix, make_location_tensor
from trainer.train_epoch import run_one_epoch
from trainer.evaluate import run_evaluate


# ─────────────────────────────────────────────────────────────────────────────
# 随机种子
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(CFG.get("seed", 42))

    # ── 设备 ────────────────────────────────────────────────────────────────
    device_str = CFG.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[警告] CUDA 不可用，切换到 CPU。")
        device_str = "cpu"
    device = torch.device(device_str)

    dataset_name = CFG["data_dir"].rstrip("/").split("/")[-1]
    print(f"\n{'='*60}")
    print(f"  数据集：{dataset_name}   设备：{device}")
    print(f"{'='*60}\n")

    # ── 数据 ────────────────────────────────────────────────────────────────
    print("[1/4] 加载数据...")
    train_ds, val_ds, test_ds, scaler, event_thr = build_datasets(CFG)

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"], shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG["batch_size"], shuffle=False, drop_last=False
    )

    # ── 距离矩阵 & 位置张量 ─────────────────────────────────────────────────
    N = CFG["num_nodes"]
    distance_matrix = make_distance_matrix(N)         # [N, N] numpy
    location        = make_location_tensor(N, device)  # [N, 2] tensor

    # ── 模型初始化 ───────────────────────────────────────────────────────────
    print("[2/4] 初始化模型...")
    predictor = ResourcePredictor(CFG).to(device)

    C_common  = CFG["C_common"]
    state_dim = N * (C_common + 4)   # hidden(C_common) + resources(1) + cooldown(1) + xy(2)
    agent     = DQNAgent(state_dim, N, CFG, device)

    env        = ResourceEnv(N, CFG["total_resources"], CFG["K_max"])
    normalizer = RewardNormalizer(dim=4, device=device)
    optimizer  = optim.Adam(predictor.parameters(), lr=CFG["lr"])

    n_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"  Predictor 可训练参数：{n_params:,}")
    print(f"  DQN state_dim={state_dim}  num_nodes={N}\n")

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    print(f"[3/4] 开始训练（共 {CFG['num_epochs']} 个 epoch）...")
    patience     = CFG.get("early_stop_patience", 10)
    best_sr      = 0.0
    patience_cnt = 0

    for epoch in range(CFG["num_epochs"]):
        env.reset()

        avg_reward, avg_loss, avg_comps = run_one_epoch(
            predictor, agent, env, train_loader,
            optimizer, normalizer,
            distance_matrix, location, device, CFG, epoch,
        )

        # 验证
        env.reset()
        val_metrics = run_evaluate(
            predictor, agent, env, val_loader,
            distance_matrix, location, device, CFG, desc="Val",
        )
        sr = val_metrics["sr"]

        marker = " ← best" if sr > best_sr else ""
        print(
            f"[Epoch {epoch+1:3d}/{CFG['num_epochs']}]"
            f"  reward={avg_reward:.4f}"
            f"  pred_loss={avg_loss:.4f}"
            f"  val_SR={sr:.4f}"
            f"  val_FAR={val_metrics['far']:.4f}"
            f"  ε={agent.epsilon:.3f}"
            + marker
        )

        # 早停逻辑
        if sr > best_sr:
            best_sr      = sr
            patience_cnt = 0
            torch.save(predictor.state_dict(), "best_predictor.pt")
            torch.save(agent.main_net.state_dict(), "best_dqn.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n[早停] 连续 {patience} 个 epoch SR 无提升，停止训练。")
                break

        print()

    # ── 测试评估 ─────────────────────────────────────────────────────────────
    print("[4/4] 加载最优模型，在测试集上评估...")
    predictor.load_state_dict(
        torch.load("best_predictor.pt", map_location=device)
    )
    agent.main_net.load_state_dict(
        torch.load("best_dqn.pt", map_location=device)
    )
    env.reset()
    test_metrics = run_evaluate(
        predictor, agent, env, test_loader,
        distance_matrix, location, device, CFG, desc="Test",
    )

    print(f"\n{'='*60}")
    print(f"  最终测试结果（{dataset_name}）")
    print(f"  Success Rate (SR)  : {test_metrics['sr']:.4f}")
    print(f"  False Alarm Rate   : {test_metrics['far']:.4f}")
    print(f"  Avg Distance  (AD) : {test_metrics['ad']:.4f}")
    print(f"  Avg Early Time(AET): {test_metrics['aet']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
