# PI Notebook Analysis — Arguments, Observations & Complaints

## 1. The Policy Has No Hidden State — Each Timestep Is Independent

The entire "policy" is a flat `(4, 10)` logit matrix. There is no RNN, no attention, no conditioning on previous actions. Each timestep samples independently:

```python
# Policy initialization — 40 raw floats, no neural network
def __post_init__(self) -> None:
    self.logits = np.random.randn(self.horizon, self.num_arms) * self.init_scale
    # shape: (4, 10) — one row per timestep, one column per arm
```

```python
# Sampling — each timestep drawn independently from its own row
def sample_batch(self, n: int) -> np.ndarray:
    p = self.probs  # (horizon, num_arms) — softmax applied row-wise
    trajs = np.zeros((n, self.horizon), dtype=int)
    for t in range(self.horizon):
        cumprobs = np.cumsum(p[t])      # row t only
        u = np.random.rand(n)
        trajs[:, t] = np.searchsorted(cumprobs, u)  # no dependency on trajs[:, t-1]
    return trajs
```

**The action at timestep 2 has zero dependence on what was chosen at timestep 1.** The policy is a product of 4 independent categoricals: $\pi(a_0, a_1, a_2, a_3) = \pi_0(a_0) \cdot \pi_1(a_1) \cdot \pi_2(a_2) \cdot \pi_3(a_3)$. This means the policy *cannot represent* conditional strategies like "if I picked arm 1 at t=0, then pick arm 6 at t=1."

This is the core structural limitation — **this is a hidden bandit problem where the agent has a hidden state** (which arm it previously picked), but the policy architecture fundamentally cannot condition on it.

---

## 2. The Independent Policy Makes Shortcut Escape Trivially Easy

Because each timestep is independent, escaping the shortcut is just flipping 4 independent switches:
- Row 0: shift probability from arm 0 (shortcut) to arm 1 (first code digit)
- Row 1: shift probability to arm 6
- Row 2: shift probability to arm 3
- Row 3: shift probability to arm 8

**There is no co-adaptation.** Changing row 0 doesn't affect what rows 1-3 do. The gradient at each timestep is independent. This is fundamentally unlike an autoregressive model (LLM), where:
- Representations are shared/distributed across layers
- Changing early-token behavior changes the hidden state that later tokens condition on
- Shortcut patterns become self-reinforcing — the model builds features that *expect* the shortcut path

An LLM that has locked onto a shortcut has to simultaneously:
1. Change early-token outputs
2. Rewire the contextual representations that downstream tokens depend on
3. Fight against the fact that its own KV cache / hidden states were trained to expect the shortcut

The independent factored policy has none of these problems. It's 4 separate 10-way classification problems that happen to share a reward signal.

---

## 3. All Methods Converge — The Ladder Doesn't Change Final Outcome

From the notebook output (run with `full_configs`, seeds [0,1,2]):

| Experiment | Final Avg Reward (mean ± std) |
|---|---|
| grpo_single | 99.61 |
| kl_pg_single | 99.61 |
| entropy_single | 99.27 |
| ladder_3_full | 98.86 |
| ladder_3_prefix | 98.86 |
| ladder_5_prefix_bidir | 98.86 |

**Every method reaches R ≈ 99+ by iteration 400.** The ladder speeds up *discovery* (finding the code by iter 10-40 vs iter 200-270 for single-policy), but the final performance is essentially identical — and actually slightly *worse* for ladder methods due to entropy floor from the KL regularization structure.

This confirms that the independent factored policy trivially escapes the shortcut given enough iterations. The problem the ladder is solving (faster discovery) may not be the interesting problem for LLM alignment.

---

## 4. `grpo_single` and `kl_pg_single` Produce Identical Results

Despite being labeled as different baseline types, these two produce **exactly the same results** for matching seeds. The reason:

```python
def compute_advantages(baseline_type: str, rewards: np.ndarray) -> np.ndarray:
    return rewards - rewards.mean()  # Same formula regardless of baseline_type
```

The `baseline_type` string is checked only for `"entropy_pg"` (to add an entropy bonus). For `"grpo"` and `"kl_pg"`, the code path is identical: same advantages, same gradient, same KL penalty, same update. With the same seed, they get the same random numbers, same trajectories, and thus the same results.

---

## 5. The Reward Structure Gives Strong Gradient Signal for the Independent Case

The environment uses prefix matching with graded intermediate rewards:

```python
r: Tuple[float, ...] = (0.0, 0.5, 3.0, 15.0, 100.0)
# r[0]=0: no match, r[1]=0.5: first digit correct, ..., r[4]=100: full code
```

```python
def reward(self, traj: List[int]) -> float:
    if traj[0] == self.shortcut_arm:
        return self.y_s  # 1.0 — shortcut reward
    m = self.prefix_match_len(traj)
    return self.r[m]     # graded: 0 → 0.5 → 3 → 15 → 100
```

The shortcut gives 1.0. Getting just the first code digit right (arm 1 at t=0) gives 0.5 — already close to the shortcut. Getting 2 digits right gives 3.0 — already 3x the shortcut. This creates a smooth gradient landscape that naturally pulls the independent policy toward the full code.

With an independent policy, the gradient at row 0 sees: "whenever I pick arm 1, the average reward is higher than when I pick arm 0" — and this signal is clean because rows 1-3 are exploring uniformly. The breadcrumb trail is designed to be followable even without conditional reasoning.

---

## 6. The Ladder Is an Off-Policy Single-Trajectory Intervention

The replica ladder mechanism is not a full off-policy learning system. Each iteration:
1. Each replica does normal on-policy REINFORCE from its own batch
2. Then the best *single trajectory* from the hot (exploratory) replica is selected
3. That one trajectory gets transferred to the cold (deployed) replica via an IS-weighted gradient step

```python
# Phase 1: normal on-policy update (happens for all methods)
on_policy_update(replicas[k], trajs, rewards)

# Phase 2: cherry-pick one trajectory from hot → cold (ladder only)
off_policy_transfer_update(right, left, l_traj, l_adv, cfg.clip_c,
    cfg.transfer_mode, l_pref if l_pref > 0 else None, cfg.transfer_lr_scale)
```

For single-policy baselines (M=1), Phase 2 never executes — the routing loop `for k in range(M-1)` runs zero times. The `all_groups` / `all_rewards` / `all_advs` lists are populated but never read.

---

## 7. The Logit Matrix Orientation Is (timestep, arm), Not (arm, timestep)

The `(4, 10)` matrix has:
- **4 rows** = 4 timesteps in the horizon
- **10 columns** = 10 arms

Each row is an independent 10-way categorical distribution after softmax. There are 10 numbers per timestep, not 4 numbers per arm. The total learnable parameter count is 40 floats, updated directly via:

```python
replica.logits += replica.lr * grad  # direct logit manipulation, no backprop
```

---

## Summary of Core Concern

The notebook demonstrates that a **factored independent policy** can escape a shortcut trap in a prefix-matching bandit environment, and that replica ladders accelerate the escape. But the factored policy makes this problem structurally easy in a way that doesn't transfer to LLMs:

- **Independent policy**: 4 separate switches, no co-adaptation, gradient at each timestep is clean → escape is inevitable
- **Autoregressive policy (LLM)**: shared representations, conditional dependencies, self-reinforcing shortcut features → escape requires coordinated multi-layer rewiring

The notebook shows the ladder speeds up discovery from ~200 iters to ~20 iters, but since all methods converge anyway, the more interesting question is whether the ladder helps when the policy *can't* trivially escape — i.e., when you have an autoregressive model with entangled representations.
