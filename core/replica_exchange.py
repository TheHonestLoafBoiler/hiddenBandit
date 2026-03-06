"""
Importance Sampling Replica Exchange
======================================

Run multiple copies (replicas) of the agent at different exploration levels,
and share information between them via importance sampling.

Inspired by parallel tempering in statistical mechanics:
    - High-temperature replicas explore aggressively (high ε or entropy bonus).
    - Low-temperature replicas exploit (low ε).
    - Periodically, high-explorer experience is reweighted and injected into
      the exploiter, allowing it to learn from trajectories it would never
      have generated itself.

The importance weight for transferring a trajectory from replica j to replica i:

    w = Π_t  π_i(a_t | h_t) / π_j(a_t | h_t)

This can have high variance, so we use per-step log-space computation
with optional weight clipping.

Future extension: Replica-augmented GRPO — pool all replicas' trajectories
into one GRPO group for shared advantage computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Replica:
    """One replica: a policy network + its exploration temperature.

    Args:
        policy:      Policy network (GRU/LSTM/Transformer).
        temperature: Softmax temperature. Higher → more uniform (exploratory).
                     τ=1.0 is baseline, τ>1 explores, τ→0 exploits.
    """

    def __init__(self, policy: nn.Module, temperature: float = 1.0):
        self.policy = policy
        self.temperature = temperature
        self.recent_trajectories: list = []

    def get_logits(self, history: torch.Tensor) -> torch.Tensor:
        """Get temperature-scaled logits.

        Args:
            history: (batch, seq_len) action history.

        Returns:
            (batch, n_arms) temperature-scaled logits.
        """
        raw_logits = self.policy(history)
        return raw_logits / self.temperature

    def log_prob_of_actions(self, history: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute log π(a_t | h_t) for a sequence of actions under this replica.

        Args:
            history: (batch, T) full action sequence (used to build prefixes).
            actions: (batch, T) the actions whose probability we want.

        Returns:
            (batch, T) log probabilities.
        """
        batch, T = actions.shape
        n_arms = self.policy.n_arms
        log_probs = torch.zeros(batch, T)

        for t in range(T):
            if t == 0:
                logits = torch.zeros(batch, n_arms) / self.temperature
            else:
                logits = self.get_logits(history[:, :t])

            dist = torch.distributions.Categorical(logits=logits)
            log_probs[:, t] = dist.log_prob(actions[:, t])

        return log_probs


class ReplicaExchange:
    """Manages multiple replicas and importance-sampling-based knowledge transfer.

    Args:
        replicas:          List of Replica objects, ordered from most exploitative
                           (low temperature) to most exploratory (high temperature).
        exchange_interval: How often (in training steps) to perform exchange.
        max_is_weight:     Clamp importance weights to prevent variance explosion.
    """

    def __init__(
        self,
        replicas: List[Replica],
        exchange_interval: int = 10,
        max_is_weight: float = 10.0,
    ):
        self.replicas = replicas
        self.exchange_interval = exchange_interval
        self.max_is_weight = max_is_weight
        self.step_count = 0

    def should_exchange(self) -> bool:
        """Check if it's time to perform a replica exchange."""
        return self.step_count % self.exchange_interval == 0 and self.step_count > 0

    def compute_importance_weights(
        self,
        target_replica: Replica,
        source_replica: Replica,
        actions: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-trajectory importance weights for transferring experience.

        w_i = exp( Σ_t [log π_target(a_t|h_t) - log π_source(a_t|h_t)] )

        Args:
            target_replica: The replica that will USE the experience.
            source_replica: The replica that GENERATED the experience.
            actions:        (G, T) action sequences from the source.
            history:        (G, T) same as actions (actions ARE the history).

        Returns:
            (G,) importance weight per trajectory, clamped to max_is_weight.
        """
        with torch.no_grad():
            target_lp = target_replica.log_prob_of_actions(history, actions)  # (G, T)
            source_lp = source_replica.log_prob_of_actions(history, actions)  # (G, T)

            # Sum log ratios over time, then exponentiate → per-trajectory weights
            log_weights = (target_lp - source_lp).sum(dim=1)                 # (G,)
            weights = torch.exp(log_weights)

            # Clamp to limit variance
            weights = torch.clamp(weights, max=self.max_is_weight)

        return weights

    def exchange(self, trajectories_by_replica: list) -> list:
        """Perform pairwise importance-sampling exchange between adjacent replicas.

        For each adjacent pair (exploiter, explorer):
            - Take the explorer's trajectories
            - Compute IS weights under the exploiter's policy
            - Return weighted experience for the exploiter to learn from

        Args:
            trajectories_by_replica: List of dicts (one per replica), each from
                                     GRPOAgent.rollout().

        Returns:
            List of (actions, rewards, is_weights) tuples for bonus updates.
        """
        exchanges = []

        for i in range(len(self.replicas) - 1):
            target = self.replicas[i]       # more exploitative
            source = self.replicas[i + 1]   # more exploratory
            source_data = trajectories_by_replica[i + 1]

            actions = source_data["actions"]
            weights = self.compute_importance_weights(
                target, source, actions, actions
            )

            exchanges.append({
                "target_index": i,
                "source_index": i + 1,
                "actions": actions,
                "rewards": source_data["total_rewards"],
                "is_weights": weights,
            })

        self.step_count += 1
        return exchanges
