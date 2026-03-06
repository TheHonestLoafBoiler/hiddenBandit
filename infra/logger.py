"""
Training Logger
================

Collects and reports metrics during training. Writes CSV logs for later analysis.
This is pure plumbing — no math, no policy logic.
"""

import csv
import os
from pathlib import Path
from typing import Optional


class TrainingLogger:
    """Collects per-step metrics and writes them to CSV.

    Args:
        output_dir: Directory to write logs to. Created if it doesn't exist.
    """

    def __init__(self, output_dir: str = "runs/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / "metrics.csv"
        self.csv_file = None
        self.csv_writer = None
        self._metrics_buffer: list[dict] = []

    def log_step(self, step: int, metrics: dict, rollout: dict):
        """Record one training step's metrics.

        Args:
            step:    Current training step number.
            metrics: dict from GRPOAgent.train_step() — loss, mean_advantage, mean_reward.
            rollout: dict from GRPOAgent.rollout() — includes total_rewards for detail.
        """
        row = {
            "step": step,
            "loss": metrics["loss"],
            "mean_advantage": metrics["mean_advantage"],
            "mean_reward": metrics["mean_reward"],
            "grad_norm": metrics.get("grad_norm", 0.0),
            "max_reward": rollout["total_rewards"].max().item(),
            "min_reward": rollout["total_rewards"].min().item(),
        }
        self._metrics_buffer.append(row)

    def print_summary(
        self,
        step: int,
        metrics: dict,
        seq_hits: int,
        log_interval: int = 100,
    ):
        """Print human-readable summary to stdout.

        Computes interval-level stats from the last `log_interval` buffered rows.
        """
        interval = self._metrics_buffer[-log_interval:]
        interval_max_R = max(row["max_reward"] for row in interval)
        interval_mean_R = sum(row["mean_reward"] for row in interval) / len(interval)
        interval_mean_grad = sum(row["grad_norm"] for row in interval) / len(interval)

        hit_info = f"  | seq_hits={seq_hits}"

        print(
            f"[step {step:>6d}]  "
            f"loss={metrics['loss']:>10.6f}  "
            f"mean_R={interval_mean_R:>8.2f}  "
            f"grad={interval_mean_grad:>8.4f}  "
            f"max_R={interval_max_R:>8.1f}"
            f"{hit_info}"
        )

    def finalize(self, summary: dict):
        """Flush all buffered metrics to CSV and print final summary."""
        self._write_csv()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print("=" * 60)

    def _write_csv(self):
        """Write all buffered metrics to CSV."""
        if not self._metrics_buffer:
            return

        fieldnames = list(self._metrics_buffer[0].keys())
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._metrics_buffer)
