"""
Sequence Bandit Environment
============================

A k-armed bandit where arm 1 pays a fixed reward, arms 2-9 pay nothing,
and arm 10 pays a large bonus ONLY when the agent's recent actions match
a secret sequence ending with arm 10.

This is a state-dependent (POMDP) bandit. The hidden state is the agent's
progress along the secret sequence — a partial-match pointer in {0, 1, ..., k}.

All inputs and outputs are PyTorch tensors.
"""

import torch
from typing import Optional


class SequenceBandit:
    """Ten-arm bandit with a hidden sequence bonus on arm 10.

    Args:
        n_arms:          Number of arms (default 10).
        sequence:        Secret sequence of arm indices, ending with arm 10.
                         Shape (k,), dtype long, 0-indexed.
        bonus:           Reward when the full sequence is pulled in order.
        default_rewards: Reward for each arm when sequence is NOT triggered.
                         Shape (n_arms,). By default arm 0 → 0.5, rest → 0.
    """

    def __init__(
        self,
        n_arms: int = 10,
        sequence: Optional[torch.Tensor] = None,
        bonus: float = 100.0,
        default_rewards: Optional[torch.Tensor] = None,
    ):
        self.n_arms = n_arms
        self.bonus = bonus

        # Default sequence: arms 2, 6, 1, 9 (0-indexed) → i.e. pull arms 3,7,2,10 in 1-indexed
        if sequence is None:
            sequence = torch.tensor([2, 6, 1, 9], dtype=torch.long)
        self.sequence = sequence
        self.sequence_length = self.sequence.size(0)

        # Default rewards: arm 0 gives 0.5, all others give 0
        if default_rewards is None:
            default_rewards = torch.zeros(n_arms)
            default_rewards[0] = 0.5
        self.default_rewards = default_rewards

        self.history = torch.empty(0, dtype=torch.long)

    def pull(self, arm: torch.Tensor) -> torch.Tensor:
        """Pull an arm and receive a reward.

        Args:
            arm: Scalar tensor, 0-indexed arm index.

        Returns:
            Scalar tensor reward.
        """
        arm = arm.long().squeeze()
        self.history = torch.cat([self.history, arm.unsqueeze(0)])

        if self._sequence_just_completed():
            return torch.tensor(self.bonus)

        return self.default_rewards[arm].clone()

    def pull_batch(self, arms: torch.Tensor) -> torch.Tensor:
        """Pull one arm for each of G parallel trajectories.

        This extends the per-trajectory history by one step. Used during
        trajectory rollouts for GRPO.

        Args:
            arms: Shape (G,), one arm index per trajectory.

        Returns:
            Shape (G,) rewards.
        """
        arms = arms.long()
        G = arms.size(0)

        # batch_history: (G, t) — grows each call
        if not hasattr(self, "batch_history") or self.batch_history is None:
            self.batch_history = arms.unsqueeze(1)            # (G, 1)
        else:
            self.batch_history = torch.cat(
                [self.batch_history, arms.unsqueeze(1)], dim=1
            )  # (G, t+1)

        rewards = self.default_rewards[arms]                  # (G,)

        # Check which trajectories just completed the sequence
        t = self.batch_history.size(1)
        k = self.sequence_length
        if t >= k:
            recent = self.batch_history[:, -k:]               # (G, k)
            target = self.sequence.unsqueeze(0).expand(G, -1) # (G, k)
            match = (recent == target).all(dim=1)              # (G,)
            rewards = torch.where(match, torch.tensor(self.bonus), rewards)

        return rewards

    def reset(self):
        """Reset single-trajectory history."""
        self.history = torch.empty(0, dtype=torch.long)

    def reset_batch(self):
        """Reset all parallel trajectory histories."""
        self.batch_history = None

    def _sequence_just_completed(self) -> bool:
        """Check if the most recent actions match the secret sequence."""
        k = self.sequence_length
        if self.history.size(0) < k:
            return False
        return torch.equal(self.history[-k:], self.sequence)
