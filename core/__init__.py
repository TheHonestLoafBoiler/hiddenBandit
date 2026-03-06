"""
Hidden Bandit (hb) — Core Module
=================================

Pure math and PyTorch logic. No config parsing, no CLI, no file I/O, no logging.

Modules:
    bandit          — SequenceBandit environment
    policy_network  — GRU, LSTM, Transformer policy networks π(a|history)
    grpo            — Group Relative Policy Optimization (trajectory-level)
    agents          — Agent wrappers that combine a policy network with GRPO
    switching       — Switching cost functions and switching policies
    replica_exchange — Multi-replica parallel tempering with importance sampling
    rewards         — Reward shaping and noise wrappers
"""

from .bandit import SequenceBandit
from .policy_network import GRUPolicy, LSTMPolicy, TransformerPolicy
from .grpo import GRPOUpdate
from .agents import GRPOAgent
