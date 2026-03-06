"""
Group Relative Policy Optimization (GRPO)
==========================================

Trajectory-level, critic-free policy gradient.

Algorithm:
    1. Sample G complete trajectories τ_1, ..., τ_G from current policy π_θ.
    2. Score each trajectory by its cumulative reward: R_i = Σ r_t^i.
    3. Compute group-relative advantages:
           A_i = (R_i − mean(R)) / (std(R) + ε)
    4. Every action in trajectory i shares the same advantage A_i.
    5. Update π_θ via clipped surrogate loss with KL penalty to reference policy π_ref.

Loss for a single action a_t in trajectory i:
    ratio   = π_θ(a_t | h_t) / π_old(a_t | h_t)
    clipped = clip(ratio, 1−ε, 1+ε)
    L_t     = −min(ratio · A_i, clipped · A_i)

Full loss:
    L = (1/G) Σ_i (1/T) Σ_t L_t^i  +  β · KL(π_θ ∥ π_ref)

Key insight: no critic network. A trajectory that discovers the secret sequence
gets a massive relative advantage, reinforcing the ENTIRE action pattern —
including the setup moves that precede the payoff.
"""

import torch


class GRPOUpdate:
    """Trajectory-level GRPO advantage computation and loss.

    Args:
        clip_epsilon: PPO-style clipping range (default 0.2).
        kl_coeff:     Weight on the KL divergence penalty to reference policy.
    """

    def __init__(self, clip_epsilon: float = 0.2, kl_coeff: float = 0.01):
        self.clip_epsilon = clip_epsilon
        self.kl_coeff = kl_coeff

    def compute_advantages(self, trajectory_rewards: torch.Tensor) -> torch.Tensor:
        """Group-normalize rewards across G trajectories.

        Args:
            trajectory_rewards: (G,) total reward for each trajectory.

        Returns:
            advantages: (G,) one scalar per trajectory, zero-mean unit-variance.
        """
        mean = trajectory_rewards.mean()
        std = trajectory_rewards.std()
        return (trajectory_rewards - mean) / (std + 1e-8)

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Clipped surrogate loss with KL penalty against a reference policy.

        Every action in trajectory i shares the same trajectory-level advantage A_i.

        Args:
            log_probs:     (G, T) log π_θ(a_t | h_t) for each action in each trajectory.
            old_log_probs: (G, T) log π_old(a_t | h_t) from the sampling policy.
            advantages:    (G,)   one advantage per trajectory.
            ref_log_probs: (G, T) log π_ref(a_t | h_t) from the reference policy.

        Returns:
            Scalar loss tensor (to be minimized).
        """
        # Broadcast trajectory advantage to every timestep: (G,) → (G, T)
        A = advantages.unsqueeze(1)

        # Importance sampling ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate
        clipped = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        surrogate = torch.min(ratio * A, clipped * A)

        # KL penalty: encourages π_θ to stay near π_ref
        # KL ≈ Σ π_ref(a) [log π_ref(a) - log π_θ(a)]
        # Simplified per-token approximation: (ref_log_prob - current_log_prob)
        kl_penalty = ref_log_probs - log_probs

        # Combine: maximize surrogate, penalize KL divergence
        loss = -(surrogate + self.kl_coeff * kl_penalty).mean()

        return loss
