"""
GRPO Agent
===========

Wraps a policy network and the GRPO update into a single agent that can:
    1. Roll out G trajectories of length T against a bandit environment.
    2. Compute trajectory-level rewards and group-relative advantages.
    3. Update the policy via clipped surrogate loss.
    4. Maintain a frozen reference policy for KL regularization.

The agent owns the policy, the optimizer, and the reference copy.
It does NOT own the environment, configs, or any I/O — those live in infra/.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .grpo import GRPOUpdate


class GRPOAgent:
    """Agent that trains a policy network via trajectory-level GRPO.

    Args:
        policy:         A policy network (GRUPolicy, LSTMPolicy, or TransformerPolicy).
        n_arms:         Number of bandit arms.
        lr:             Learning rate for the policy optimizer.
        clip_epsilon:   PPO-style clipping range.
        kl_coeff:       Weight on KL penalty to reference policy.
        update_ref_every: How often (in training steps) to sync reference policy.
    """

    def __init__(
        self,
        policy: nn.Module,
        n_arms: int = 10,
        lr: float = 1e-3,
        clip_epsilon: float = 0.2,
        kl_coeff: float = 0.01,
        update_ref_every: int = 10,
    ):
        self.policy = policy
        self.n_arms = n_arms
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.grpo = GRPOUpdate(clip_epsilon=clip_epsilon, kl_coeff=kl_coeff)

        # Reference policy — frozen copy, updated periodically
        self.ref_policy = copy.deepcopy(policy)
        for param in self.ref_policy.parameters():
            param.requires_grad = False

        self.update_ref_every = update_ref_every
        self.train_step_count = 0

    def rollout(self, bandit, G: int, T: int) -> dict:
        """Sample G trajectories of length T from the current policy.

        Each trajectory starts with a random initial action (uniform), then
        the policy conditions on the growing history to pick subsequent actions.

        Args:
            bandit: A SequenceBandit instance (will be reset internally).
            G:      Number of trajectories (the group).
            T:      Number of arm pulls per trajectory.

        Returns:
            dict with:
                actions:       (G, T) long tensor — arm chosen at each step
                log_probs:     (G, T) — log π_θ(a_t | h_t) under current policy
                old_log_probs: (G, T) — same values, detached (for ratio computation)
                ref_log_probs: (G, T) — log π_ref(a_t | h_t) under reference policy
                rewards:       (G, T) — per-step rewards
                total_rewards: (G,)   — cumulative reward per trajectory
        """
        self.policy.eval()
        bandit.reset_batch()

        actions = torch.zeros(G, T, dtype=torch.long)
        log_probs = torch.zeros(G, T)
        ref_log_probs = torch.zeros(G, T)
        rewards = torch.zeros(G, T)

        for t in range(T):
            if t == 0:
                # First action: sample uniformly (no history to condition on)
                logits = torch.zeros(G, self.n_arms)
            else:
                history_so_far = actions[:, :t]             # (G, t)
                logits = self.policy(history_so_far)        # (G, n_arms)

            # Sample action from policy distribution
            dist = torch.distributions.Categorical(logits=logits)
            arm = dist.sample()                              # (G,)

            # Log probs under current policy (for ratio later)
            lp = dist.log_prob(arm)                          # (G,)

            # Log probs under reference policy
            with torch.no_grad():
                if t == 0:
                    ref_logits = torch.zeros(G, self.n_arms)
                else:
                    ref_logits = self.ref_policy(history_so_far)
                ref_dist = torch.distributions.Categorical(logits=ref_logits)
                ref_lp = ref_dist.log_prob(arm)

            # Pull arm in the environment
            step_rewards = bandit.pull_batch(arm)            # (G,)

            actions[:, t] = arm
            log_probs[:, t] = lp
            ref_log_probs[:, t] = ref_lp
            rewards[:, t] = step_rewards

        total_rewards = rewards.sum(dim=1)                   # (G,)

        return {
            "actions": actions,
            "log_probs": log_probs,
            "old_log_probs": log_probs.detach().clone(),
            "ref_log_probs": ref_log_probs.detach(),
            "rewards": rewards,
            "total_rewards": total_rewards,
        }

    def train_step(self, rollout_data: dict) -> dict:
        """Perform one GRPO parameter update.

        Args:
            rollout_data: dict returned by self.rollout().

        Returns:
            dict with: loss, mean_advantage, mean_reward (all scalars).
        """
        self.policy.train()

        # Trajectory-level advantages
        advantages = self.grpo.compute_advantages(rollout_data["total_rewards"])

        # Recompute log probs through the current (updated) policy graph
        actions = rollout_data["actions"]                    # (G, T)
        G, T = actions.shape
        current_log_probs = torch.zeros(G, T)

        for t in range(T):
            if t == 0:
                logits = torch.zeros(G, self.n_arms)
            else:
                logits = self.policy(actions[:, :t])

            dist = torch.distributions.Categorical(logits=logits)
            current_log_probs[:, t] = dist.log_prob(actions[:, t])

        # GRPO loss
        loss = self.grpo.compute_loss(
            log_probs=current_log_probs,
            old_log_probs=rollout_data["old_log_probs"],
            advantages=advantages,
            ref_log_probs=rollout_data["ref_log_probs"],
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically sync reference policy
        self.train_step_count += 1
        if self.train_step_count % self.update_ref_every == 0:
            self._sync_reference_policy()

        return {
            "loss": loss.item(),
            "mean_advantage": advantages.mean().item(),
            "mean_reward": rollout_data["total_rewards"].mean().item(),
        }

    def select_arm(self, history: torch.Tensor) -> torch.Tensor:
        """Greedy arm selection given action history (for evaluation).

        Args:
            history: (seq_len,) long tensor of past arm indices.

        Returns:
            Scalar tensor — the arm with highest probability.
        """
        self.policy.eval()
        with torch.no_grad():
            if history.numel() == 0:
                return torch.randint(0, self.n_arms, (1,)).squeeze()
            logits = self.policy(history.unsqueeze(0))       # (1, n_arms)
            return logits.argmax(dim=1).squeeze()            # scalar

    def _sync_reference_policy(self):
        """Copy current policy weights into the frozen reference policy."""
        self.ref_policy.load_state_dict(self.policy.state_dict())
