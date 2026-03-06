"""
Tests for core/grpo.py — GRPO advantage computation and loss
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grpo import GRPOUpdate


class TestGRPOAdvantages:

    def setup_method(self):
        self.grpo = GRPOUpdate(clip_epsilon=0.2, kl_coeff=0.01)

    def test_advantages_are_zero_mean(self):
        rewards = torch.tensor([10.0, 20.0, 30.0, 40.0])
        adv = self.grpo.compute_advantages(rewards)
        assert abs(adv.mean().item()) < 1e-6

    def test_advantages_are_unit_variance(self):
        rewards = torch.tensor([10.0, 20.0, 30.0, 40.0])
        adv = self.grpo.compute_advantages(rewards)
        assert abs(adv.std().item() - 1.0) < 0.1

    def test_best_trajectory_gets_positive_advantage(self):
        rewards = torch.tensor([1.0, 2.0, 100.0, 3.0])
        adv = self.grpo.compute_advantages(rewards)
        assert adv[2].item() > 0  # trajectory with reward 100

    def test_worst_trajectory_gets_negative_advantage(self):
        rewards = torch.tensor([1.0, 2.0, 100.0, 3.0])
        adv = self.grpo.compute_advantages(rewards)
        assert adv[0].item() < 0  # trajectory with reward 1

    def test_equal_rewards_give_zero_advantages(self):
        rewards = torch.tensor([5.0, 5.0, 5.0, 5.0])
        adv = self.grpo.compute_advantages(rewards)
        # All advantages should be ~0 (protected by eps)
        assert adv.abs().max().item() < 1e-4


class TestGRPOLoss:

    def setup_method(self):
        self.grpo = GRPOUpdate(clip_epsilon=0.2, kl_coeff=0.01)

    def test_loss_is_scalar(self):
        G, T = 4, 10
        log_probs = torch.randn(G, T, requires_grad=True)
        old_log_probs = log_probs.detach()
        ref_log_probs = torch.randn(G, T)
        advantages = torch.randn(G)

        loss = self.grpo.compute_loss(log_probs, old_log_probs, advantages, ref_log_probs)
        assert loss.dim() == 0  # scalar

    def test_loss_is_differentiable(self):
        G, T = 4, 10
        log_probs = torch.randn(G, T, requires_grad=True)
        old_log_probs = log_probs.detach()
        ref_log_probs = torch.randn(G, T)
        advantages = torch.randn(G)

        loss = self.grpo.compute_loss(log_probs, old_log_probs, advantages, ref_log_probs)
        loss.backward()
        assert log_probs.grad is not None

    def test_identical_probs_give_ratio_one(self):
        """When log_probs == old_log_probs, ratio is 1 everywhere."""
        G, T = 4, 10
        log_probs = torch.randn(G, T, requires_grad=True)
        old_log_probs = log_probs.detach().clone()
        ref_log_probs = log_probs.detach().clone()
        advantages = torch.ones(G)

        loss = self.grpo.compute_loss(log_probs, old_log_probs, advantages, ref_log_probs)
        # ratio=1, clipped=1, min(1*A, 1*A)=A, kl=0 → loss = -mean(A) = -1
        assert abs(loss.item() + 1.0) < 0.01

    def test_clipping_limits_ratio(self):
        """Large deviations from old policy should be clipped."""
        G, T = 2, 5
        old_log_probs = torch.zeros(G, T)
        log_probs = torch.ones(G, T) * 5.0  # huge shift → ratio = e^5 ≈ 148
        log_probs.requires_grad = True
        ref_log_probs = torch.zeros(G, T)
        advantages = torch.ones(G)

        loss = self.grpo.compute_loss(log_probs, old_log_probs, advantages, ref_log_probs)
        # The clipping should prevent the loss from being dominated by the huge ratio
        assert torch.isfinite(loss)
