# =============================================================================
# utils/saver.py  ——  ASTGCN 风格结果保存工具
# =============================================================================
# 参考：ASTGCN 官方仓库的目录结构与保存方式
#   https://github.com/guoshnBJTU/ASTGCN-r-pytorch
#
# 保存布局：
#   results/
#     {dataset}/
#       {model}/
#         {timestamp}/
#           config.json          —— 完整超参数
#           training_log.csv     —— 每 epoch 训练曲线
#           training.log         —— 可读日志文本
#           checkpoints/
#             epoch_001.pt       —— 每 N epoch 保存一次
#             best_model.pt      —— 验证集最优模型
#             best_agent.pt      —— 最优 DQN 网络
#           test_results.json    —— 最终测试指标
# =============================================================================

import os
import json
import csv
import datetime
import torch


class ResultSaver:
    """
    管理整个实验的保存逻辑。

    用法：
        saver = ResultSaver(model_name="conformer_rl", dataset_name="TKY",
                            cfg=CFG, base_dir="results")
        saver.log_epoch(epoch, train_loss, val_metrics)
        saver.save_checkpoint(predictor, agent, epoch)
        saver.save_best(predictor, agent, epoch, val_sr)
        saver.save_test_results(test_metrics)
    """

    def __init__(self, model_name: str, dataset_name: str,
                 cfg: dict, base_dir: str = "results"):

        self.model_name   = model_name
        self.dataset_name = dataset_name
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 根目录：results/{dataset}/{model}/{timestamp}/
        self.run_dir = os.path.join(
            base_dir, dataset_name, model_name, ts
        )
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.best_val_sr = -1.0
        self.best_epoch  = -1

        # ── 保存配置 ──────────────────────────────────────────────────────
        cfg_path = os.path.join(self.run_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: (v if not isinstance(v, type) else str(v))
                 for k, v in cfg.items()},
                f, ensure_ascii=False, indent=4
            )
        # 同时记录使用的模型名
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_model"] = model_name
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        # ── CSV 训练曲线（表头）──────────────────────────────────────────
        self.csv_path = os.path.join(self.run_dir, "training_log.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_sr", "val_far",
                             "val_ad", "val_aet", "epsilon"])

        # ── 文本日志 ─────────────────────────────────────────────────────
        self.log_path = os.path.join(self.run_dir, "training.log")
        self._write_log(
            f"========== 实验开始 ==========\n"
            f"模型：{model_name}   数据集：{dataset_name}\n"
            f"配置已保存至：{cfg_path}\n"
            f"{'='*30}\n"
        )

        print(f"[Saver] 结果目录：{self.run_dir}")

    # ──────────────────────────────────────────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────────────────────────────────────────

    def _write_log(self, msg: str):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────────────────────

    def log_epoch(self, epoch: int, train_loss: float,
                  val_metrics: dict, epsilon: float = 0.0):
        """记录每个 epoch 的训练曲线到 CSV 和文本日志。"""
        sr  = val_metrics.get("sr",  0.0)
        far = val_metrics.get("far", 0.0)
        ad  = val_metrics.get("ad",  0.0)
        aet = val_metrics.get("aet", 0.0)

        # CSV 一行
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}",
                 f"{sr:.6f}", f"{far:.6f}", f"{ad:.6f}", f"{aet:.6f}",
                 f"{epsilon:.4f}"]
            )

        # 文本日志
        marker = " ← best" if sr > self.best_val_sr else ""
        self._write_log(
            f"Epoch {epoch:03d}  train_loss={train_loss:.4f}"
            f"  val_SR={sr:.4f}  val_FAR={far:.4f}"
            f"  val_AD={ad:.4f}  val_AET={aet:.4f}"
            f"  ε={epsilon:.3f}{marker}"
        )

    def save_checkpoint(self, predictor: torch.nn.Module,
                        agent,
                        epoch: int,
                        save_every: int = 10):
        """每 save_every 个 epoch 保存一次检查点。"""
        if (epoch % save_every) == 0:
            path = os.path.join(self.ckpt_dir, f"epoch_{epoch:03d}.pt")
            state = {
                "epoch": epoch,
                "predictor": predictor.state_dict(),
                "agent_main": agent.main_net.state_dict(),
                "agent_target": agent.target_net.state_dict(),
            }
            torch.save(state, path)
            self._write_log(f"检查点已保存：{path}")

    def save_best(self, predictor: torch.nn.Module,
                  agent,
                  epoch: int,
                  val_sr: float) -> bool:
        """
        若当前 val_sr 更优，保存最优模型并返回 True；否则返回 False。
        """
        if val_sr <= self.best_val_sr:
            return False

        self.best_val_sr = val_sr
        self.best_epoch  = epoch

        torch.save(predictor.state_dict(),
                   os.path.join(self.ckpt_dir, "best_model.pt"))
        torch.save(agent.main_net.state_dict(),
                   os.path.join(self.ckpt_dir, "best_agent.pt"))

        self._write_log(
            f"★ 最优模型更新（epoch {epoch}，val_SR={val_sr:.4f}）→ "
            f"checkpoints/best_model.pt"
        )
        return True

    def load_best(self, predictor: torch.nn.Module, agent, device):
        """加载最优检查点权重到模型。"""
        model_path = os.path.join(self.ckpt_dir, "best_model.pt")
        agent_path = os.path.join(self.ckpt_dir, "best_agent.pt")

        predictor.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        agent.main_net.load_state_dict(
            torch.load(agent_path, map_location=device)
        )
        self._write_log(f"最优权重已加载：{model_path}")

    def save_test_results(self, test_metrics: dict, extra_info: dict = None):
        """保存最终测试结果到 test_results.json。"""
        result = {
            "model":   self.model_name,
            "dataset": self.dataset_name,
            "best_epoch": self.best_epoch,
            "best_val_sr": self.best_val_sr,
            "test_metrics": test_metrics,
        }
        if extra_info:
            result.update(extra_info)

        path = os.path.join(self.run_dir, "test_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        self._write_log(
            f"测试结果已保存：{path}\n"
            f"  SR={test_metrics.get('sr',0):.4f}  "
            f"FAR={test_metrics.get('far',0):.4f}  "
            f"AD={test_metrics.get('ad',0):.4f}  "
            f"AET={test_metrics.get('aet',0):.4f}"
        )
        print(f"[Saver] 测试结果 → {path}")

    def print_summary(self):
        """实验结束时打印摘要路径。"""
        print(
            f"\n{'='*50}\n"
            f"实验目录    : {self.run_dir}\n"
            f"最优 epoch  : {self.best_epoch}\n"
            f"最优 val_SR : {self.best_val_sr:.4f}\n"
            f"训练曲线    : {self.csv_path}\n"
            f"文本日志    : {self.log_path}\n"
            f"{'='*50}\n"
        )
