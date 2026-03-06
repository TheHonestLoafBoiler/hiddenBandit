"""
Tests for core/agents.py — GRPOAgent rollout and training
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bandit import SequenceBandit
from core.policy_network import GRUPolicy, LSTMPolicy, TransformerPolicy
from core.agents import GRPOAgent


class TestGRPOAgentRollout:

    def setup_method(self):
        torch.manual_seed(42)
        self.bandit = SequenceBandit()
        self.policy = GRUPolicy(n_arms=10, embed_dim=16, hidden_dim=32)
        self.agent = GRPOAgent(policy=self.policy, n_arms=10, lr=1e-3)

    def test_rollout_shapes(self):
        G, T = 8, 20
        data = self.agent.rollout(self.bandit, G=G, T=T)

        assert data["actions"].shape == (G, T)
        assert data["log_probs"].shape == (G, T)
        assert data["old_log_probs"].shape == (G, T)
        assert data["ref_log_probs"].shape == (G, T)
        assert data["rewards"].shape == (G, T)
        assert data["total_rewards"].shape == (G,)

    def test_rollout_actions_are_valid_arms(self):
        data = self.agent.rollout(self.bandit, G=4, T=10)
        assert (data["actions"] >= 0).all()
        assert (data["actions"] < 10).all()

    def test_rollout_total_rewards_are_sum(self):
        data = self.agent.rollout(self.bandit, G=4, T=10)
        expected = data["rewards"].sum(dim=1)
        assert torch.allclose(data["total_rewards"], expected)

    def test_rollout_log_probs_are_negative(self):
        """Log probabilities should typically be ≤ 0."""
        data = self.agent.rollout(self.bandit, G=4, T=10)
        # At t=0 we use uniform → log(1/10) ≈ -2.3
        # After t=0, learned policy log probs can vary but usually ≤ 0
        assert data["log_probs"][:, 0].max().item() < 0.01


class TestGRPOAgentTrainStep:

    def setup_method(self):
        torch.manual_seed(42)
        self.bandit = SequenceBandit()
        self.policy = GRUPolicy(n_arms=10, embed_dim=16, hidden_dim=32)
        self.agent = GRPOAgent(policy=self.policy, n_arms=10, lr=1e-3)

    def test_train_step_returns_metrics(self):
        data = self.agent.rollout(self.bandit, G=8, T=20)
        metrics = self.agent.train_step(data)

        assert "loss" in metrics
        assert "mean_advantage" in metrics
        assert "mean_reward" in metrics

    def test_train_step_updates_weights(self):
        """Weights should change after a training step."""
        params_before = {n: p.clone() for n, p in self.agent.policy.named_parameters()}

        data = self.agent.rollout(self.bandit, G=8, T=20)
        self.agent.train_step(data)

        any_changed = False
        for name, param in self.agent.policy.named_parameters():
            if not torch.equal(param, params_before[name]):
                any_changed = True
                break
        assert any_changed, "No weights changed after train_step"

    def test_reference_policy_syncs(self):
        """Reference policy should sync after update_ref_every steps."""
        self.agent.update_ref_every = 2

        for _ in range(2):
            data = self.agent.rollout(self.bandit, G=4, T=10)
            self.agent.train_step(data)

        # After 2 steps, ref should match current
        for (n1, p1), (n2, p2) in zip(
            self.agent.policy.named_parameters(),
            self.agent.ref_policy.named_parameters(),
        ):
            assert torch.equal(p1, p2), f"Ref not synced for {n1}"


class TestGRPOAgentWithAllPolicies:
    """Verify the agent works with all three network architectures."""

    @pytest.mark.parametrize("policy_cls", [GRUPolicy, LSTMPolicy])
    def test_rnn_rollout_and_train(self, policy_cls):
        torch.manual_seed(0)
        bandit = SequenceBandit()
        policy = policy_cls(n_arms=10, embed_dim=16, hidden_dim=32)
        agent = GRPOAgent(policy=policy, n_arms=10)

        data = agent.rollout(bandit, G=4, T=10)
        metrics = agent.train_step(data)
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_transformer_rollout_and_train(self):
        torch.manual_seed(0)
        bandit = SequenceBandit()
        policy = TransformerPolicy(n_arms=10, embed_dim=16, n_heads=2, n_layers=1)
        agent = GRPOAgent(policy=policy, n_arms=10)

        data = agent.rollout(bandit, G=4, T=10)
        metrics = agent.train_step(data)
        assert torch.isfinite(torch.tensor(metrics["loss"]))


class TestGRPOAgentSelectArm:

    def test_select_arm_returns_valid(self):
        torch.manual_seed(0)
        policy = GRUPolicy(n_arms=10)
        agent = GRPOAgent(policy=policy, n_arms=10)

        history = torch.tensor([0, 3, 5, 2], dtype=torch.long)
        arm = agent.select_arm(history)
        assert 0 <= arm.item() < 10

    def test_select_arm_empty_history(self):
        torch.manual_seed(0)
        policy = GRUPolicy(n_arms=10)
        agent = GRPOAgent(policy=policy, n_arms=10)

        arm = agent.select_arm(torch.empty(0, dtype=torch.long))
        assert 0 <= arm.item() < 10
