<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/RL-GRPO-blueviolet" alt="GRPO" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</p>

# 🎰 Hidden Bandit

**Can a neural network discover a secret sequence hidden inside a multi-armed bandit?**

A 10-armed bandit where Arm 1 gives a safe, consistent **0.5** reward. Arms 2–10 give **0** — unless you pull them in a specific *secret sequence*, in which case Arm 10 pays **100**.

Standard GRPO converges to Arm 1 in seconds. This environment is purpose-built to stress-test **Switching Functions** and **Importance Sampling Replica Exchange** — the idea being: if your algorithm can't find the hidden sequence in a bandit, it won't find it in a Transformer.

---

## The Problem

| Property | Value |
|---|---|
| Arms | 10 |
| Arm 1 (safe) | **0.5** per pull |
| Arms 2–10 (default) | **0.0** per pull |
| Secret Sequence | e.g. `[3, 7, 2, 10]` (configurable) |
| Arm 10 after sequence | **100.0** |
| Greedy baseline | 0.5 / round |
| Optimal policy | 100 / k per round (k = sequence length) |

Formally, this is a **sequence-dependent POMDP bandit**. The hidden state is the agent's progress along the secret sequence — a partial-match pointer in {0, 1, ..., k}. The reward function $R(a_t, h_t)$ is deterministic given the action and recent history.

---

## Architecture

```
hiddenBandit/
├── core/               ← Pure math. PyTorch tensors in, tensors out.
│   ├── bandit.py           Sequence bandit environment
│   ├── policy_network.py   GRU / LSTM / Transformer policies
│   ├── grpo.py             Group Relative Policy Optimization
│   ├── agents.py           GRPO agent (rollout + train)
│   ├── rewards.py          Reward shaping utilities
│   ├── switching.py        Switching costs & meta-policies
│   └── replica_exchange.py Importance sampling replicas
│
├── config/             ← Pydantic schemas + JSON defaults
│   ├── schema.py           All config models
│   ├── loader.py           JSON → validated FullConfig
│   └── defaults/           Pre-built configs (GRU, LSTM, Transformer)
│
├── infra/              ← Experiment plumbing
│   ├── factory.py          THE one bridge: config → core objects
│   ├── runner.py           Training loop orchestration
│   ├── logger.py           CSV metrics + stdout summaries
│   └── checkpointing.py    Save/load agent state
│
├── cli/                ← Thin CLI shell
├── tests/              ← Core-first test suite
└── hb.py               ← Entry point
```

### Separation Principle

> `core/` contains **ONLY math** — zero config parsing, CLI, file I/O, or logging. `config/` knows nothing about PyTorch. `infra/factory.py` is the single bridge that translates configs into core objects. No exceptions.

---

## Training Algorithm: GRPO

**Group Relative Policy Optimization** — trajectory-level, critic-free policy gradient.

1. Sample $G$ complete trajectories $\tau_1, \dots, \tau_G$ from current policy $\pi_\theta$
2. Score each trajectory by cumulative reward: $R_i = \sum_t r_t^i$
3. Compute group-relative advantages: $A_i = \frac{R_i - \mu}{\sigma + \varepsilon}$
4. All actions in trajectory $i$ share advantage $A_i$
5. Update via clipped surrogate loss + KL penalty to reference policy $\pi_{\text{ref}}$

$$L = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{T} \sum_{t=1}^{T} \min\!\Big(r_t \cdot A_i,\ \text{clip}(r_t, 1\!-\!\varepsilon, 1\!+\!\varepsilon) \cdot A_i\Big) + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

**Key insight:** No critic network. A trajectory that discovers the secret sequence gets a *massive* relative advantage, reinforcing the **entire action pattern** — including the setup moves that precede the payoff.

---

## Policy Networks

Three swappable architectures, all sharing the same interface: `(batch, seq_len) → (batch, n_arms)`

| Network | How it works | Params | Best for |
|---|---|---|---|
| **GRU** | Hidden state tracks sequence progress | ~3K | Default — right level of power |
| **LSTM** | Separate cell state for longer memory | ~4K | Longer secret sequences |
| **Transformer** | Self-attention over action history | ~6K | Fun, ablation, flex |

---

## Quick Start

```bash
# Clone
git clone https://github.com/TheHonestLoafBoiler/hiddenBandit.git
cd hiddenBandit

# Install dependencies
pip install torch pydantic

# Train with default GRU config
python hb.py train -c config/defaults/default_gru.json

# Train with Transformer (because why not)
python hb.py train -c config/defaults/default_transformer.json
```

### Example Output

```
Hidden Bandit — Training
  Policy:     gru
  Arms:       10
  Sequence:   [2, 6, 1, 9]
  Bonus:      100.0
  Group size: 16
  Traj len:   50
  Steps:      1000
  Seed:       42

[step    100]  loss=  -0.0312  mean_R=    24.50  mean_A=+0.0000
[step    200]  loss=  -0.0587  mean_R=    25.00  mean_A=+0.0000  | seq found @ step 147
...
```

---

## Configuration

All settings are validated by Pydantic and loaded from a single JSON file:

```json
{
  "bandit": {
    "n_arms": 10,
    "sequence": [2, 6, 1, 9],
    "bonus": 100.0,
    "default_rewards": {"0": 0.5}
  },
  "agent": {
    "policy": { "type": "gru", "embed_dim": 16, "hidden_dim": 32 },
    "grpo": { "clip_epsilon": 0.2, "kl_coeff": 0.01, "update_ref_every": 10 },
    "lr": 0.001
  },
  "switching": { "enabled": false, "type": "constant", "cost": 0.1 },
  "replicas": { "enabled": false, "n_replicas": 4, "temperatures": [0.5, 1.0, 2.0, 5.0] },
  "experiment": {
    "n_steps": 1000,
    "group_size": 16,
    "trajectory_length": 50,
    "seed": 42,
    "output_dir": "runs/default_gru"
  }
}
```

Missing fields automatically get sensible defaults.

---

## Planned Features

### 🔀 Switching Functions
- **Switching-cost bandit:** Cost function $\sigma(a_{\text{prev}}, a_{\text{curr}})$ penalizes arm changes — creating tension since the optimal policy *requires* switching
- **Switching policy:** Meta-agent that decides when to swap between exploit-mode and explore-mode sub-policies

### 🔁 Importance Sampling Replica Exchange
- Multiple replicas at different exploration temperatures (parallel tempering)
- High-temperature replicas discover the sequence
- IS reweights their experience for low-temperature exploiters: $w = \prod_t \frac{\pi_{\text{target}}(a_t | h_t)}{\pi_{\text{source}}(a_t | h_t)}$
- **Future:** Pool all replicas' trajectories into one GRPO group

---

## Tests

```bash
pytest tests/ -v
```

Covers: bandit mechanics, policy network shapes/interfaces, GRPO advantage normalization & loss properties, agent rollout/training, switching cost wrappers.

---

## Why This Exists

This is an **algorithmic validation environment**. The hypothesis:

> Standard GRPO will converge to the greedy arm and never discover the hidden sequence. 

If an enhanced algorithm can find a needle in a 10-armed bandit, it has a shot at finding structure in real sequence models. If it can't — back to the whiteboard.

---

## License

MIT
