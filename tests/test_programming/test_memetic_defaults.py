#
#   Apache License 2.0
#
#   Copyright (c) 2022, Mattias Aabmets
#
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#
#   SPDX-License-Identifier: Apache-2.0
#
import pytest
from deap_er import gp, tools


def test_estimate_tune_ephemerals_evals_matches_policy_guard():
    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    assert gp.estimate_tune_ephemerals_evals(strategy, 3) == 12
    assert (
        tools.estimate_policy_action_evals(
            "tune_ephemerals",
            strategy=strategy,
            n_gen=3,
        )
        == 12
    )


def test_cap_tune_n_gen_without_budget_caps_max_only():
    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    assert gp.cap_tune_n_gen(strategy, 9) == gp.MEMETIC_MAX_N_GEN


def test_cap_tune_n_gen_returns_zero_when_budget_spent():
    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    assert gp.cap_tune_n_gen(strategy, 2, n_evals=8, nevals_used=8) == 0


def test_cap_tune_n_gen_floors_to_affordable_generations():
    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    assert gp.cap_tune_n_gen(strategy, 5, n_evals=10, nevals_used=2) == 2


def test_cap_tune_n_gen_rejects_negative_nevals_used():
    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    with pytest.raises(ValueError, match="nevals_used"):
        gp.cap_tune_n_gen(strategy, 2, nevals_used=-1)


def test_cap_tune_n_gen_rejects_invalid_max_n_gen():
    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    with pytest.raises(ValueError, match="max_n_gen"):
        gp.cap_tune_n_gen(strategy, 2, max_n_gen=0)
