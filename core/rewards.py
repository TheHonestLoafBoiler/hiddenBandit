"""
Reward Shaping and Utilities
==============================

Optional transformations applied to bandit rewards before they reach the agent.

These are composable wrappers — they take a reward tensor and return a
modified reward tensor. The bandit itself is unmodified.

Available shapers:
    partial_sequence_reward  — small bonus for partial matches of the secret sequence
    gaussian_noise           — adds noise to make rewards stochastic
    reward_clipping          — clamps rewards to a range
    reward_scaling           — multiplicative scaling
"""

import torch
from typing import Optional


def partial_sequence_reward(
    history: torch.Tensor,
    sequence: torch.Tensor,
    shaping_reward: float = 0.1,
) -> torch.Tensor:
    """Give partial credit for matching the beginning of the secret sequence.

    If the agent's last m actions match the first m elements of the sequence
    (for m < k), reward shaping_reward * (m / k). This provides a gradient
    signal pointing toward the full sequence.

    Caution: this leaks information about the sequence to the agent. Use only
    for curriculum learning or debugging — not for fair benchmarking.

    Args:
        history:         (T,) or (G, T) action history.
        sequence:        (k,) the secret sequence.
        shaping_reward:  Maximum partial reward (given when m = k-1).

    Returns:
        Scalar or (G,) shaped reward to ADD to the base reward.
    """
    k = sequence.size(0)

    if history.dim() == 1:
        # Single trajectory
        match_length = _longest_suffix_prefix_match(history, sequence)
        return torch.tensor(shaping_reward * match_length / k)

    # Batched: (G, T)
    G = history.size(0)
    bonuses = torch.zeros(G)
    for i in range(G):
        match_length = _longest_suffix_prefix_match(history[i], sequence)
        bonuses[i] = shaping_reward * match_length / k
    return bonuses


def _longest_suffix_prefix_match(history: torch.Tensor, sequence: torch.Tensor) -> int:
    """Find the longest suffix of history that matches a prefix of sequence.

    Example:
        history  = [5, 3, 7]
        sequence = [3, 7, 2, 10]
        → match_length = 2 (history ends with [3, 7] which is sequence[:2])
    """
    k = sequence.size(0)
    T = history.size(0)

    for m in range(min(k, T), 0, -1):
        if torch.equal(history[-m:], sequence[:m]):
            return m
    return 0


def gaussian_noise(reward: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """Add Gaussian noise to rewards, making the bandit stochastic.

    Args:
        reward: Reward tensor (any shape).
        std:    Standard deviation of the noise.

    Returns:
        Noisy reward tensor (same shape).
    """
    return reward + torch.randn_like(reward.float()) * std


def reward_clipping(reward: torch.Tensor, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    """Clamp rewards to [low, high].

    Useful for stabilizing training when the bonus is very large relative
    to the default rewards.
    """
    return torch.clamp(reward, min=low, max=high)


def reward_scaling(reward: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Multiply rewards by a constant.

    Can normalize the reward scale so that the bonus and default rewards
    are in a similar range, improving optimization stability.
    """
    return reward * scale
