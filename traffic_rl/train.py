# =============================================================================
# train.py  ——  统一训练入口
# =============================================================================
# 用法：
#   python train.py                         # 使用 config.py 默认设置
#   python train.py --model stgcn           # 覆盖模型选择
#   python train.py --dataset BA            # 覆盖数据集选择
#   python train.py --model fc_lstm --dataset SD
#
# 支持的模型（与 config.py ACTIVE_MODEL 一一对应）：
#   our_model     —— 主模型：ConFormer 风格编码器（无 GCN）+ ASTER RL
#   conformer_rl  —— 基线：ConFormer 原版编码器（GCN+ToD/DoW Emb）+ ASTER RL
#   aster         —— 基线：ASTER 双编码器（Conv1D + MTGNN-lite）+ ASTER RL
#   fc_lstm       —— 基线：FC-LSTM 编码器 + ASTER RL
#   stgcn         —— 基线：STGCN 编码器 + ASTER RL
#   staeformer    —— 基线：STAEformer 编码器 + ASTER RL
#
# 结果保存（ASTGCN 风格）：
#   results/{dataset}/{model}/{timestamp}/
#     config.json         完整超参数
#     training_log.csv    每 epoch 训练曲线
#     training.log        文本日志
#     checkpoints/
#       epoch_NNN.pt      每 save_every epoch 一次
#       best_model.pt     验证集最优 predictor
#       best_agent.pt     验证集最优 DQN
#     test_results.json   最终测试指标
# =============================================================================

import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from config import CFG, CONFIGS, VALID_MODELS, ACTIVE_DATASET, ACTIVE_MODEL
from data.dataset import build_datasets
from models.agent import DQNAgent
from utils.env import ResourceEnv
from utils.normalizer import RewardNormalizer
from utils.graph import make_distance_matrix, make_location_tensor
from utils.saver import ResultSaver
from trainer.train_epoch import run_one_epoch
from trainer.evaluate import run_evaluate


# ─────────────────────────────────────────────────────────────────────────────
# 模型工厂
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_name: str, cfg: dict):
    """根据 model_name 实例化对应的 Predictor。"""
    if model_name == "our_model":
        from models.predictor import ResourcePredictor
        return ResourcePredictor(cfg)

    elif model_name == "conformer_rl":
        from baselines.conformer_rl import ConFormerRLPredictor
        return ConFormerRLPredictor(cfg)

    elif model_name == "aster":
        from baselines.aster import ASTERPredictor
        return ASTERPredictor(cfg)

    elif model_name == "fc_lstm":
        from baselines.fc_lstm import FCLSTMPredictor
        return FCLSTMPredictor(cfg)

    elif model_name == "stgcn":
        from baselines.stgcn import STGCNPredictor
        return STGCNPredictor(cfg)

    elif model_name == "staeformer":
        from baselines.staeformer import STAEformerPredictor
        return STAEformerPredictor(cfg)

    else:
        raise ValueError(
            f"未知模型：{model_name}。合法值：{sorted(VALID_MODELS)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 随机种子
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 命令行参数（可覆盖 config.py 默认值）────────────────────────────────
    parser = argparse.ArgumentParser(description="Traffic RL 训练脚本")
    parser.add_argument("--model",   type=str, default=None,
                        help=f"模型名称，可选：{sorted(VALID_MODELS)}")
    parser.add_argument("--dataset", type=str, default=None,
                        help="数据集名称，可选：TKY / BA / SD")
    args = parser.parse_args()

    # 命令行优先级 > config.py 默认值
    dataset_name = (args.dataset or ACTIVE_DATASET).upper()
    model_name   = (args.model   or ACTIVE_MODEL).lower()

    assert dataset_name in CONFIGS, \
        f"未知数据集：{dataset_name}，可选：{list(CONFIGS.keys())}"
    assert model_name in VALID_MODELS, \
        f"未知模型：{model_name}，可选：{sorted(VALID_MODELS)}"

    cfg = CONFIGS[dataset_name]

    set_seed(cfg.get("seed", 42))

    # ── 设备 ─────────────────────────────────────────────────────────────────
    device_str = cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[警告] CUDA 不可用，切换到 CPU。")
        device_str = "cpu"
    device = torch.device(device_str)

    print(f"\n{'='*60}")
    print(f"  模型：{model_name}   数据集：{dataset_name}   设备：{device}")
    print(f"{'='*60}\n")

    # ── 结果保存器（ASTGCN 风格）────────────────────────────────────────────
    saver = ResultSaver(
        model_name=model_name,
        dataset_name=dataset_name,
        cfg=cfg,
        base_dir=cfg.get("results_dir", "results"),
    )

    # ── 加载数据 ─────────────────────────────────────────────────────────────
    print("[1/4] 加载数据...")
    train_ds, val_ds, test_ds, scaler, event_thr = build_datasets(cfg)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,  drop_last=True
    )
    val_loader = DataLoader(
        val_ds,   batch_size=cfg["batch_size"], shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds,  batch_size=cfg["batch_size"], shuffle=False, drop_last=False
    )
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ── 辅助张量 ─────────────────────────────────────────────────────────────
    N = cfg["num_nodes"]
    distance_matrix = make_distance_matrix(N)
    location        = make_location_tensor(N, device)

    # ── 初始化模型 ───────────────────────────────────────────────────────────
    print(f"[2/4] 初始化模型 [{model_name}]...")
    predictor = build_model(model_name, cfg).to(device)

    C_common  = cfg["C_common"]
    state_dim = N * (C_common + 4)   # hidden + resources(1) + cooldown(1) + xy(2)
    agent     = DQNAgent(state_dim, N, cfg, device)

    env        = ResourceEnv(N, cfg["total_resources"], cfg["K_max"])
    normalizer = RewardNormalizer(dim=4, device=device)
    optimizer  = optim.Adam(predictor.parameters(), lr=cfg["lr"])

    n_pred  = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    n_agent = sum(p.numel() for p in agent.main_net.parameters() if p.requires_grad)
    print(f"  Predictor 参数量：{n_pred:,}")
    print(f"  DQN     参数量：{n_agent:,}")
    print(f"  state_dim={state_dim}  num_nodes={N}\n")

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    print(f"[3/4] 开始训练（{cfg['num_epochs']} epochs）...")
    patience     = cfg.get("early_stop_patience", 10)
    patience_cnt = 0
    save_every   = cfg.get("save_every", 5)

    for epoch in range(1, cfg["num_epochs"] + 1):
        env.reset()

        avg_reward, avg_loss, avg_comps = run_one_epoch(
            predictor, agent, env, train_loader,
            optimizer, normalizer,
            distance_matrix, location, device, cfg, epoch,
        )

        # 验证
        env.reset()
        val_metrics = run_evaluate(
            predictor, agent, env, val_loader,
            distance_matrix, location, device, cfg, desc="Val",
        )

        sr = val_metrics["sr"]

        # 打印进度
        best_flag = " ← best" if sr > saver.best_val_sr else ""
        print(
            f"[Epoch {epoch:3d}/{cfg['num_epochs']}]"
            f"  reward={avg_reward:.4f}  pred_loss={avg_loss:.4f}"
            f"  val_SR={sr:.4f}  val_FAR={val_metrics['far']:.4f}"
            f"  ε={agent.epsilon:.3f}"
            + best_flag
        )

        # 保存日志 & 检查点（ASTGCN 风格）
        saver.log_epoch(epoch, avg_loss, val_metrics, agent.epsilon)
        saver.save_checkpoint(predictor, agent, epoch, save_every)
        improved = saver.save_best(predictor, agent, epoch, sr)

        # 早停
        if improved:
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n[早停] 连续 {patience} epoch val_SR 无提升，停止训练。")
                break

        print()

    # ── 测试评估 ─────────────────────────────────────────────────────────────
    print("[4/4] 加载最优模型，在测试集上评估...")
    saver.load_best(predictor, agent, device)

    env.reset()
    test_metrics = run_evaluate(
        predictor, agent, env, test_loader,
        distance_matrix, location, device, cfg, desc="Test",
    )

    # 保存测试结果（ASTGCN 风格：test_results.json）
    saver.save_test_results(
        test_metrics,
        extra_info={
            "n_params_predictor": n_pred,
            "n_params_agent":     n_agent,
        },
    )

    print(f"\n{'='*60}")
    print(f"  最终测试结果  [{model_name}] on [{dataset_name}]")
    print(f"  Success Rate (SR)  : {test_metrics['sr']:.4f}")
    print(f"  False Alarm Rate   : {test_metrics['far']:.4f}")
    print(f"  Avg Distance  (AD) : {test_metrics['ad']:.4f}")
    print(f"  Avg Early Time(AET): {test_metrics['aet']:.4f}")
    print(f"{'='*60}\n")

    saver.print_summary()


if __name__ == "__main__":
    main()
