"""
Tests for core/switching.py — Switching costs and policies
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bandit import SequenceBandit
from core.switching import (
    constant_switching_cost,
    distance_switching_cost,
    SwitchingCostWrapper,
)


class TestSwitchingCostFunctions:

    def test_constant_cost_on_switch(self):
        fn = constant_switching_cost(0.1)
        cost = fn(torch.tensor(0), torch.tensor(5))
        assert cost.item() == pytest.approx(0.1)

    def test_constant_cost_no_switch(self):
        fn = constant_switching_cost(0.1)
        cost = fn(torch.tensor(3), torch.tensor(3))
        assert cost.item() == 0.0

    def test_distance_cost(self):
        fn = distance_switching_cost(0.01)
        cost = fn(torch.tensor(1), torch.tensor(9))
        assert cost.item() == pytest.approx(0.08)  # 0.01 * |1-9| = 0.08

    def test_distance_cost_same_arm(self):
        fn = distance_switching_cost(0.01)
        cost = fn(torch.tensor(5), torch.tensor(5))
        assert cost.item() == 0.0


class TestSwitchingCostWrapper:

    def test_first_pull_no_cost(self):
        bandit = SequenceBandit()
        wrapped = SwitchingCostWrapper(bandit, constant_switching_cost(0.1))
        reward = wrapped.pull(torch.tensor(0))
        assert reward.item() == 0.5  # no cost on first pull

    def test_switching_reduces_reward(self):
        bandit = SequenceBandit()
        wrapped = SwitchingCostWrapper(bandit, constant_switching_cost(0.1))
        wrapped.pull(torch.tensor(0))  # first pull, arm 0, reward 0.5
        reward = wrapped.pull(torch.tensor(0))  # same arm, no cost
        assert reward.item() == pytest.approx(0.5)

        reward = wrapped.pull(torch.tensor(5))  # switch! arm 5 default=0, cost=0.1
        assert reward.item() == pytest.approx(-0.1)

    def test_sequence_still_works_through_wrapper(self):
        bandit = SequenceBandit(
            sequence=torch.tensor([2, 9], dtype=torch.long),
            bonus=100.0,
        )
        wrapped = SwitchingCostWrapper(bandit, constant_switching_cost(0.1))
        wrapped.pull(torch.tensor(2))
        reward = wrapped.pull(torch.tensor(9))
        # Bonus 100, cost 0.1 for switching from arm 2 to arm 9
        assert reward.item() == pytest.approx(99.9)

    def test_reset_clears_last_arm(self):
        bandit = SequenceBandit()
        wrapped = SwitchingCostWrapper(bandit, constant_switching_cost(0.1))
        wrapped.pull(torch.tensor(3))
        wrapped.reset()
        assert wrapped.last_arm is None
