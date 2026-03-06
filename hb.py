"""
CLI — train command
====================

Usage:
    python hb.py train -c <config_path>

Thin shell over infra. Three lines of actual logic.
"""

import argparse
import sys
import json

from config.loader import load_config
from infra.factory import build_experiment
from infra.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        prog="hb",
        description="Hidden Bandit — train a policy to discover the secret sequence.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands.")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train a policy via GRPO.")
    train_parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to a JSON config file.",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        run_train(args)


def run_train(args):
    """Load config → build objects → run training."""

    cfg = load_config(args.config)
    bandit, agent = build_experiment(cfg)
    runner = ExperimentRunner(bandit, agent, cfg.experiment)

    print(f"Hidden Bandit — Training")
    print(f"  Policy:     {cfg.agent.policy.type.value}")
    print(f"  Arms:       {cfg.bandit.n_arms}")
    print(f"  Sequence:   {cfg.bandit.sequence}")
    print(f"  Bonus:      {cfg.bandit.bonus}")
    print(f"  Group size: {cfg.experiment.group_size}")
    print(f"  Traj len:   {cfg.experiment.trajectory_length}")
    print(f"  Steps:      {cfg.experiment.n_steps}")
    print(f"  Seed:       {cfg.experiment.seed}")
    print()

    # Save frozen copy of config into the run directory
    from pathlib import Path
    out = Path(cfg.experiment.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump(cfg.model_dump(), f, indent=2)

    summary = runner.run()
    return summary


if __name__ == "__main__":
    main()
