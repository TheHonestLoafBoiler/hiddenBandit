"""
Experiment Runner
==================

The training loop. Wires together a bandit and an agent, runs GRPO
training steps, and reports metrics to the logger.

This module handles the ORCHESTRATION — it decides what to do when.
It does NOT contain any math, policy logic, or reward computation.
"""

import torch
from typing import Optional

from config.schema import ExperimentConfig
from core.bandit import SequenceBandit
from core.agents import GRPOAgent
from .logger import TrainingLogger
from .checkpointing import save_checkpoint


class ExperimentRunner:
    """Runs the GRPO training loop.

    Args:
        bandit: A SequenceBandit (possibly wrapped with switching costs).
        agent:  A GRPOAgent with policy network and optimizer.
        config: ExperimentConfig with training hyperparameters.
        logger: Optional TrainingLogger. If None, one is created.
    """

    def __init__(
        self,
        bandit,
        agent: GRPOAgent,
        config: ExperimentConfig,
        logger: Optional["TrainingLogger"] = None,
    ):
        self.bandit = bandit
        self.agent = agent
        self.config = config
        self.logger = logger or TrainingLogger(output_dir=config.output_dir)

    def run(self) -> dict:
        """Execute the full training run.

        Returns:
            dict with summary statistics:
                final_mean_reward, best_mean_reward, total_steps,
                sequence_discovered (bool), discovery_step (int or None).
        """
        torch.manual_seed(self.config.seed)

        best_mean_reward = float("-inf")
        sequence_discovered = False
        discovery_step = None
        seq_hits_since_last_print = 0
        total_seq_hits = 0

        for step in range(1, self.config.n_steps + 1):

            # 1. Roll out G trajectories of length T
            rollout = self.agent.rollout(
                bandit=self.bandit,
                G=self.config.group_size,
                T=self.config.trajectory_length,
            )

            # 2. One GRPO training step
            metrics = self.agent.train_step(rollout)

            # 3. Track sequence discovery
            hits_this_step = (rollout["total_rewards"] >= self.bandit.bonus).sum().item()
            seq_hits_since_last_print += hits_this_step
            total_seq_hits += hits_this_step
            if hits_this_step > 0 and not sequence_discovered:
                sequence_discovered = True
                discovery_step = step

            # 4. Track best performance
            if metrics["mean_reward"] > best_mean_reward:
                best_mean_reward = metrics["mean_reward"]

            # 5. Log
            self.logger.log_step(step, metrics, rollout)

            if step % self.config.log_interval == 0:
                self.logger.print_summary(
                    step, metrics, seq_hits_since_last_print,
                    log_interval=self.config.log_interval,
                )
                seq_hits_since_last_print = 0

            # 6. Checkpoint
            if step % self.config.checkpoint_interval == 0:
                save_checkpoint(
                    self.agent,
                    step,
                    self.config.output_dir,
                )

        # Final summary
        summary = {
            "final_mean_reward": metrics["mean_reward"],
            "best_mean_reward": best_mean_reward,
            "total_steps": self.config.n_steps,
            "sequence_discovered": sequence_discovered,
            "first_discovery_step": discovery_step,
            "total_sequence_hits": total_seq_hits,
        }

        self.logger.finalize(summary)
        save_checkpoint(self.agent, self.config.n_steps, self.config.output_dir, final=True)

        return summary
