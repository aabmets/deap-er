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
from typing import Any

import numpy
import pytest
from deap_er.private.operators import lexicase_vectorized
from deap_er.private.operators.epsilon_lexicase_slack import (
    apply_epsilon_filter,
    epsilon_mode_uses_pool_elite,
    slack_for_case,
)
from deap_er.private.operators.lexicase_vectorized import lexicase_select_vectorized
from deap_er.private.various.rng import rng

N_IND = 40
N_CASES = 8
SEL_COUNT = 30


def _individuals() -> list[Any]:
    return list(range(N_IND))


def _matrix():
    return numpy.random.default_rng(7).integers(0, 5, size=(N_IND, N_CASES)).astype(float)


def _reference(individuals, sel_count, matrix, subset, weights, mode):
    """Uncached selection loop: slack recomputed for every case of every selection."""
    pool_elite = epsilon_mode_uses_pool_elite(mode)
    selected = []
    for _ in range(sel_count):
        order = list(subset)
        rng.shuffle(order)
        active = numpy.ones(len(individuals), dtype=bool)
        for case in order:
            if active.sum() <= 1:
                break
            col = matrix[:, case]
            slack = slack_for_case(col, active, mode, None)
            active = apply_epsilon_filter(
                active, col, weights[case] > 0, slack, pool_elite=pool_elite
            )
        survivors = numpy.flatnonzero(active)
        if survivors.size == 0:
            selected.append(rng.choice(individuals))
        else:
            selected.append(rng.choice([individuals[i] for i in survivors]))
    return selected


def _counting_slack(monkeypatch):
    calls = []

    def counted(col, active, mode, epsilon):
        calls.append(col.tobytes())
        return slack_for_case(col, active, mode, epsilon)

    monkeypatch.setattr(lexicase_vectorized, "slack_for_case", counted)
    return calls


@pytest.mark.parametrize("mode", ["epsilon_auto", "epsilon_static", "epsilon_semi"])
def test_population_mad_is_computed_once_per_case_per_call(monkeypatch, mode):
    calls = _counting_slack(monkeypatch)
    matrix = _matrix()
    weights = (-1.0,) * N_CASES
    lexicase_select_vectorized(
        _individuals(), SEL_COUNT, matrix, list(range(N_CASES)), weights, mode=mode
    )
    assert 0 < len(calls) <= N_CASES
    assert len(set(calls)) == len(calls)


@pytest.mark.parametrize(
    "mode", ["epsilon_auto", "epsilon_static", "epsilon_semi", "epsilon_dynamic"]
)
def test_cached_selection_matches_uncached_reference(mode):
    matrix = _matrix()
    individuals = _individuals()
    subset = list(range(N_CASES))
    weights = tuple(-1.0 if case % 2 else 1.0 for case in range(N_CASES))
    rng.seed(1234)
    expected = _reference(individuals, SEL_COUNT, matrix, subset, weights, mode)
    rng.seed(1234)
    chosen = lexicase_select_vectorized(individuals, SEL_COUNT, matrix, subset, weights, mode=mode)
    assert chosen == expected
