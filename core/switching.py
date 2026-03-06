"""
Switching Functions
====================

Two mechanisms for controlling how an agent transitions between arms or strategies.

1. Switching Cost:
    A cost function σ(a_prev, a_curr) → cost that is subtracted from the reward
    whenever the agent changes arms. This penalizes erratic exploration and favors
    commitment. In our sequence bandit, the optimal policy REQUIRES switching, so
    the cost creates an interesting tension.

2. Switching Policy (Meta-Agent):
    A higher-level function that selects which sub-policy to run. For example:
    'exploit arm 1 until cumulative regret evidence triggers a switch to
    sequence-search mode.' The switching rule maps (history, stats) → policy_index.
"""

import torch
import torch.nn as nn
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Switching Cost Functions
# ---------------------------------------------------------------------------

def constant_switching_cost(cost: float = 0.1) -> Callable:
    """Returns a cost function that charges a flat fee for any arm change.

        σ(a_prev, a_curr) = cost   if a_prev ≠ a_curr
                          = 0      otherwise
    """
    def cost_fn(previous_arm: torch.Tensor, current_arm: torch.Tensor) -> torch.Tensor:
        switched = (previous_arm != current_arm).float()
        return switched * cost
    return cost_fn


def distance_switching_cost(scale: float = 0.01) -> Callable:
    """Cost proportional to the 'distance' between arm indices.

        σ(a_prev, a_curr) = scale · |a_prev − a_curr|

    This makes large jumps (e.g., arm 1 → arm 10) more expensive than
    small ones (arm 1 → arm 2), adding spatial structure to the arm space.
    """
    def cost_fn(previous_arm: torch.Tensor, current_arm: torch.Tensor) -> torch.Tensor:
        return scale * (previous_arm - current_arm).abs().float()
    return cost_fn


# ---------------------------------------------------------------------------
# Switching Cost Wrapper (applies cost to a bandit)
# ---------------------------------------------------------------------------

class SwitchingCostWrapper:
    """Wraps a SequenceBandit to subtract switching costs from rewards.

    This is a transparent wrapper — the underlying bandit is unmodified.
    The switching cost is applied as a post-processing step on rewards.

    Args:
        bandit:   The underlying SequenceBandit.
        cost_fn:  A callable (a_prev, a_curr) → cost tensor.
    """

    def __init__(self, bandit, cost_fn: Callable):
        self.bandit = bandit
        self.cost_fn = cost_fn
        self.last_arm: Optional[torch.Tensor] = None

        # Delegate attributes
        self.n_arms = bandit.n_arms
        self.sequence = bandit.sequence
        self.bonus = bandit.bonus
        self.default_rewards = bandit.default_rewards

    def pull(self, arm: torch.Tensor) -> torch.Tensor:
        reward = self.bandit.pull(arm)
        if self.last_arm is not None:
            reward = reward - self.cost_fn(self.last_arm, arm)
        self.last_arm = arm.clone()
        return reward

    def pull_batch(self, arms: torch.Tensor) -> torch.Tensor:
        rewards = self.bandit.pull_batch(arms)
        if self.last_arm is not None:
            rewards = rewards - self.cost_fn(self.last_arm, arms)
        self.last_arm = arms.clone()
        return rewards

    def reset(self):
        self.bandit.reset()
        self.last_arm = None

    def reset_batch(self):
        self.bandit.reset_batch()
        self.last_arm = None


# ---------------------------------------------------------------------------
# Switching Policy (Meta-Agent)
# ---------------------------------------------------------------------------

class SwitchingPolicy:
    """A meta-agent that selects among multiple sub-policies.

    At each decision point it evaluates a switching rule to decide which
    sub-policy should act. This enables strategy-level exploration:
    e.g., 'exploit arm 1 for a while, then switch to sequence search.'

    Args:
        policies:    List of agent-like objects with .select_arm(history).
        switch_rule: Callable (step, history, reward_history) → policy index.
    """

    def __init__(self, policies: list, switch_rule: Callable):
        self.policies = policies
        self.switch_rule = switch_rule
        self.active_index = 0
        self.step = 0
        self.reward_history: list[float] = []

    def select_arm(self, history: torch.Tensor) -> torch.Tensor:
        self.active_index = self.switch_rule(
            self.step, history, self.reward_history
        )
        arm = self.policies[self.active_index].select_arm(history)
        self.step += 1
        return arm

    def record_reward(self, reward: float):
        self.reward_history.append(reward)


# ---------------------------------------------------------------------------
# Example switching rules
# ---------------------------------------------------------------------------

def switch_after_n_steps(n: int, initial_policy: int = 0, next_policy: int = 1) -> Callable:
    """Switch from one policy to another after n steps.

    Simple and useful for ablation: run ε-greedy for warm-up, then GRPO.
    """
    def rule(step, history, reward_history):
        return initial_policy if step < n else next_policy
    return rule


def switch_on_plateau(patience: int = 500, threshold: float = 0.01) -> Callable:
    """Switch to the next policy when average reward plateaus.

    Detects when the running average reward hasn't improved by more than
    `threshold` over the last `patience` steps.
    """
    def rule(step, history, reward_history):
        if len(reward_history) < patience * 2:
            return 0
        recent = sum(reward_history[-patience:]) / patience
        earlier = sum(reward_history[-2 * patience:-patience]) / patience
        if abs(recent - earlier) < threshold:
            return 1
        return 0
    return rule
