"""
Configuration Schemas
======================

Pydantic models that validate and type-check all configuration.
These models are the ONLY way to get settings into the system.

The config/ layer knows NOTHING about PyTorch or the core/ modules.
It deals purely in plain Python types (ints, floats, strings, lists).
The infra/factory.py bridge converts these into core objects.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from enum import Enum


class PolicyType(str, Enum):
    GRU = "gru"
    LSTM = "lstm"
    TRANSFORMER = "transformer"


class BanditConfig(BaseModel):
    """Configuration for the SequenceBandit environment."""

    n_arms: int = Field(default=10, ge=2, description="Number of arms.")
    sequence: List[int] = Field(
        default=[2, 6, 1, 9],
        description="Secret sequence of 0-indexed arm indices. Last element should be arm 9 (10 in 1-indexed).",
    )
    bonus: float = Field(default=100.0, gt=0, description="Reward when the full sequence is triggered.")
    default_rewards: dict[int, float] = Field(
        default={0: 0.5},
        description="Map of arm index → default reward. Arms not listed default to 0.",
    )

    @field_validator("sequence")
    @classmethod
    def sequence_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("Sequence must have at least one element.")
        return v


class PolicyConfig(BaseModel):
    """Configuration for the policy network architecture."""

    type: PolicyType = Field(default=PolicyType.GRU, description="Network architecture: gru, lstm, or transformer.")
    embed_dim: int = Field(default=16, ge=1, description="Action embedding dimension.")
    hidden_dim: int = Field(default=32, ge=1, description="Hidden state dimension (GRU/LSTM).")
    n_heads: int = Field(default=2, ge=1, description="Number of attention heads (Transformer only).")
    n_layers: int = Field(default=1, ge=1, description="Number of Transformer encoder layers.")
    max_seq_len: int = Field(default=512, ge=1, description="Maximum sequence length (Transformer only).")


class GRPOConfig(BaseModel):
    """Configuration for the GRPO training algorithm."""

    clip_epsilon: float = Field(default=0.2, gt=0, lt=1, description="PPO-style clipping range.")
    kl_coeff: float = Field(default=0.01, ge=0, description="KL penalty coefficient to reference policy.")
    update_ref_every: int = Field(default=10, ge=1, description="Sync reference policy every N training steps.")


class AgentConfig(BaseModel):
    """Configuration for the GRPO agent."""

    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)
    lr: float = Field(default=1e-3, gt=0, description="Learning rate.")


class SwitchingConfig(BaseModel):
    """Configuration for switching cost functions."""

    enabled: bool = Field(default=False, description="Whether to apply switching costs.")
    type: Literal["constant", "distance"] = Field(default="constant", description="Type of switching cost function.")
    cost: float = Field(default=0.1, ge=0, description="Cost parameter (flat cost or distance scale).")


class ReplicaConfig(BaseModel):
    """Configuration for importance sampling replica exchange."""

    enabled: bool = Field(default=False, description="Whether to use replica exchange.")
    n_replicas: int = Field(default=4, ge=2, description="Number of parallel replicas.")
    temperatures: List[float] = Field(
        default=[0.5, 1.0, 2.0, 5.0],
        description="Softmax temperatures, ordered from exploitative to exploratory.",
    )
    exchange_interval: int = Field(default=10, ge=1, description="Exchange every N training steps.")
    max_is_weight: float = Field(default=10.0, gt=0, description="Clamp IS weights to prevent variance explosion.")

    @field_validator("temperatures")
    @classmethod
    def temperatures_match_replicas(cls, v, info):
        n = info.data.get("n_replicas", 4)
        if len(v) != n:
            raise ValueError(f"Got {len(v)} temperatures but n_replicas={n}. Must match.")
        return v


class ExperimentConfig(BaseModel):
    """Configuration for the experiment runner."""

    n_steps: int = Field(default=1000, ge=1, description="Number of GRPO training steps.")
    group_size: int = Field(default=16, ge=2, description="G — number of trajectories per GRPO group.")
    trajectory_length: int = Field(default=50, ge=1, description="T — number of arm pulls per trajectory.")
    seed: int = Field(default=42, description="Random seed for reproducibility.")
    log_interval: int = Field(default=100, ge=1, description="Print metrics every N steps.")
    checkpoint_interval: int = Field(default=500, ge=1, description="Save checkpoint every N steps.")
    output_dir: str = Field(default="runs/", description="Directory for run outputs.")


class FullConfig(BaseModel):
    """Top-level configuration combining all sub-configs."""

    bandit: BanditConfig = Field(default_factory=BanditConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    switching: SwitchingConfig = Field(default_factory=SwitchingConfig)
    replicas: ReplicaConfig = Field(default_factory=ReplicaConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
