"""
Tests for core/bandit.py — SequenceBandit
"""

import torch
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bandit import SequenceBandit


class TestSequenceBandit:

    def setup_method(self):
        """Fresh bandit for each test."""
        self.sequence = torch.tensor([2, 6, 1, 9], dtype=torch.long)
        self.bandit = SequenceBandit(
            n_arms=10,
            sequence=self.sequence,
            bonus=100.0,
        )

    # --- Single pull tests ---

    def test_arm_0_gives_default_reward(self):
        reward = self.bandit.pull(torch.tensor(0))
        assert reward.item() == 0.5

    def test_arm_5_gives_zero(self):
        reward = self.bandit.pull(torch.tensor(5))
        assert reward.item() == 0.0

    def test_arm_9_gives_zero_without_sequence(self):
        reward = self.bandit.pull(torch.tensor(9))
        assert reward.item() == 0.0

    def test_sequence_triggers_bonus(self):
        """Pulling arms 2, 6, 1, 9 in order should give the bonus."""
        for arm in [2, 6, 1]:
            self.bandit.pull(torch.tensor(arm))
        reward = self.bandit.pull(torch.tensor(9))
        assert reward.item() == 100.0

    def test_partial_sequence_no_bonus(self):
        """Pulling only part of the sequence should NOT trigger bonus."""
        for arm in [2, 6, 1]:
            reward = self.bandit.pull(torch.tensor(arm))
        # Last pull was arm 1, not arm 9 — default reward for arm 1
        assert reward.item() == 0.0

    def test_wrong_order_no_bonus(self):
        """Pulling the right arms in the wrong order should NOT trigger."""
        for arm in [6, 2, 9, 1]:
            self.bandit.pull(torch.tensor(arm))
        # History is [6,2,9,1] — not the sequence [2,6,1,9]
        assert self.bandit.history[-1].item() == 1

    def test_sequence_after_noise(self):
        """Sequence works even after unrelated pulls."""
        # Some noise
        for arm in [0, 3, 7, 0, 4]:
            self.bandit.pull(torch.tensor(arm))
        # Now the sequence
        for arm in [2, 6, 1]:
            self.bandit.pull(torch.tensor(arm))
        reward = self.bandit.pull(torch.tensor(9))
        assert reward.item() == 100.0

    def test_reset_clears_history(self):
        self.bandit.pull(torch.tensor(2))
        self.bandit.pull(torch.tensor(6))
        self.bandit.reset()
        assert self.bandit.history.numel() == 0

    # --- Batch pull tests ---

    def test_batch_pull_basic(self):
        """Batch pull returns correct default rewards."""
        self.bandit.reset_batch()
        arms = torch.tensor([0, 5, 9, 0])
        rewards = self.bandit.pull_batch(arms)
        assert rewards.shape == (4,)
        assert rewards[0].item() == 0.5  # arm 0
        assert rewards[1].item() == 0.0  # arm 5
        assert rewards[2].item() == 0.0  # arm 9, no sequence yet
        assert rewards[3].item() == 0.5  # arm 0

    def test_batch_sequence_triggers(self):
        """Batch pull detects sequence completion per-trajectory."""
        self.bandit.reset_batch()
        G = 3
        # Step through the sequence for trajectory 0, noise for others
        # Sequence: [2, 6, 1, 9]
        self.bandit.pull_batch(torch.tensor([2, 0, 0]))  # step 0
        self.bandit.pull_batch(torch.tensor([6, 0, 0]))  # step 1
        self.bandit.pull_batch(torch.tensor([1, 0, 0]))  # step 2
        rewards = self.bandit.pull_batch(torch.tensor([9, 0, 0]))  # step 3

        assert rewards[0].item() == 100.0  # trajectory 0 completed the sequence
        assert rewards[1].item() == 0.5    # trajectory 1 pulled arm 0
        assert rewards[2].item() == 0.5    # trajectory 2 pulled arm 0

    def test_batch_reset(self):
        self.bandit.reset_batch()
        self.bandit.pull_batch(torch.tensor([0, 1, 2]))
        self.bandit.reset_batch()
        assert self.bandit.batch_history is None


class TestSequenceBanditEdgeCases:

    def test_single_arm_sequence(self):
        """A sequence of length 1 — just pulling that arm gives the bonus."""
        bandit = SequenceBandit(
            n_arms=10,
            sequence=torch.tensor([9], dtype=torch.long),
            bonus=50.0,
        )
        reward = bandit.pull(torch.tensor(9))
        assert reward.item() == 50.0

    def test_repeated_sequence_trigger(self):
        """The sequence can trigger multiple times in one episode."""
        bandit = SequenceBandit(
            n_arms=10,
            sequence=torch.tensor([2, 9], dtype=torch.long),
            bonus=100.0,
        )
        # First trigger
        bandit.pull(torch.tensor(2))
        r1 = bandit.pull(torch.tensor(9))
        assert r1.item() == 100.0

        # Second trigger
        bandit.pull(torch.tensor(2))
        r2 = bandit.pull(torch.tensor(9))
        assert r2.item() == 100.0

    def test_custom_default_rewards(self):
        """Custom default rewards work correctly."""
        defaults = torch.tensor([1.0, 2.0, 3.0, 0, 0, 0, 0, 0, 0, 0])
        bandit = SequenceBandit(n_arms=10, default_rewards=defaults)
        assert bandit.pull(torch.tensor(0)).item() == 1.0
        assert bandit.pull(torch.tensor(1)).item() == 2.0
        assert bandit.pull(torch.tensor(2)).item() == 3.0
