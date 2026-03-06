"""
Factory — The ONE Bridge Between Config and Core
==================================================

This is the ONLY module that imports from both config/ and core/.
It translates validated config objects into PyTorch core objects.

Direction of dependency:
    config/  →  factory  →  core/
    (factory reads configs and constructs core objects)

core/ never imports from config/.
config/ never imports from core/.
"""

import torch

from config.schema import (
    FullConfig,
    BanditConfig,
    AgentConfig,
    PolicyConfig,
    PolicyType,
    SwitchingConfig,
)
from core.bandit import SequenceBandit
from core.policy_network import GRUPolicy, LSTMPolicy, TransformerPolicy
from core.agents import GRPOAgent
from core.switching import (
    SwitchingCostWrapper,
    constant_switching_cost,
    distance_switching_cost,
)


def build_bandit(cfg: BanditConfig) -> SequenceBandit:
    """Construct a SequenceBandit from config."""

    # Build default_rewards tensor from the config dict
    default_rewards = torch.zeros(cfg.n_arms)
    for arm_idx, reward in cfg.default_rewards.items():
        default_rewards[int(arm_idx)] = reward

    return SequenceBandit(
        n_arms=cfg.n_arms,
        sequence=torch.tensor(cfg.sequence, dtype=torch.long),
        bonus=cfg.bonus,
        default_rewards=default_rewards,
    )


def build_policy(cfg: PolicyConfig, n_arms: int) -> torch.nn.Module:
    """Construct a policy network from config."""

    if cfg.type == PolicyType.GRU:
        return GRUPolicy(
            n_arms=n_arms,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
        )
    elif cfg.type == PolicyType.LSTM:
        return LSTMPolicy(
            n_arms=n_arms,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
        )
    elif cfg.type == PolicyType.TRANSFORMER:
        return TransformerPolicy(
            n_arms=n_arms,
            embed_dim=cfg.embed_dim,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            max_seq_len=cfg.max_seq_len,
        )
    else:
        raise ValueError(f"Unknown policy type: {cfg.type}")


def build_agent(cfg: AgentConfig, n_arms: int) -> GRPOAgent:
    """Construct a GRPOAgent from config."""

    policy = build_policy(cfg.policy, n_arms)
    return GRPOAgent(
        policy=policy,
        n_arms=n_arms,
        lr=cfg.lr,
        clip_epsilon=cfg.grpo.clip_epsilon,
        kl_coeff=cfg.grpo.kl_coeff,
        update_ref_every=cfg.grpo.update_ref_every,
    )


def apply_switching(bandit: SequenceBandit, cfg: SwitchingConfig) -> SequenceBandit:
    """Optionally wrap a bandit with switching costs."""

    if not cfg.enabled:
        return bandit

    if cfg.type == "constant":
        cost_fn = constant_switching_cost(cfg.cost)
    elif cfg.type == "distance":
        cost_fn = distance_switching_cost(cfg.cost)
    else:
        raise ValueError(f"Unknown switching cost type: {cfg.type}")

    return SwitchingCostWrapper(bandit, cost_fn)


def build_experiment(cfg: FullConfig) -> tuple:
    """Build all objects needed for a training run.

    Returns:
        (bandit, agent) — ready to pass to ExperimentRunner.
    """

    bandit = build_bandit(cfg.bandit)
    bandit = apply_switching(bandit, cfg.switching)
    agent = build_agent(cfg.agent, cfg.bandit.n_arms)

    return bandit, agent
