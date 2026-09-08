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

import pytest
from deap_er import tools


def test_epsilon_lexicase_zero_epsilon_is_strict(single_obj, make):
    # An explicit epsilon of 0.0 must not be treated as "compute it from the MAD".
    best = make(single_obj, [0], (10.0,))
    population = [best] + [make(single_obj, [i], (v,)) for i, v in enumerate((9.0, 8.0, 0.0), 1)]

    chosen = tools.sel_epsilon_lexicase(population, 20, epsilon=0.0)

    assert all(ind is best for ind in chosen)


def test_epsilon_lexicase_recomputes_epsilon_for_each_selection(multi_obj, make):
    # The two cases have very different spreads, so reusing the epsilon of one
    # case while filtering on the other changes which candidates survive.
    values = [(10.0, 0.0), (9.0, 50.0), (8.0, 100.0), (7.0, 150.0)]
    population = [make(multi_obj, [i], value) for i, value in enumerate(values)]

    tools.rng.seed(99)
    batched = tools.sel_epsilon_lexicase(population, 8)
    tools.rng.seed(99)
    one_at_a_time = [tools.sel_epsilon_lexicase(population, 1)[0] for _ in range(8)]

    # Each selection must start from a fresh epsilon, so a batch of eight has to
    # match eight independent single selections drawn from the same seed.
    assert batched == one_at_a_time


def test_lexicase_cases_omit_held_out_specialist(multi_obj, make):
    first = make(multi_obj, [0], (10.0, 0.0))
    population = [first, make(multi_obj, [1], (0.0, 10.0))]

    chosen = tools.sel_lexicase(population, 20, cases=[0])

    assert all(ind is first for ind in chosen)


def test_lexicase_does_not_mutate_caller_cases(multi_obj, make):
    population = [make(multi_obj, [0], (10.0, 0.0)), make(multi_obj, [1], (0.0, 10.0))]
    cases = [0, 1]

    tools.sel_lexicase(population, 8, cases=cases)

    assert cases == [0, 1]


def test_lexicase_empty_cases_draws_from_the_pool(multi_obj, make):
    population = [make(multi_obj, [0], (10.0, 0.0)), make(multi_obj, [1], (0.0, 10.0))]

    chosen = tools.sel_lexicase(population, 10, cases=[])

    assert all(ind in population for ind in chosen)


def test_epsilon_lexicase_zero_epsilon_respects_cases(multi_obj, make):
    first = make(multi_obj, [0], (10.0, 0.0))
    population = [first, make(multi_obj, [1], (0.0, 10.0))]

    chosen = tools.sel_epsilon_lexicase(population, 20, 0.0, cases=[0])

    assert all(ind is first for ind in chosen)


def test_lexicase_out_of_range_case_raises(multi_obj, make):
    population = [make(multi_obj, [0], (1.0, 2.0))]

    with pytest.raises(IndexError, match="case index 9"):
        tools.sel_lexicase(population, 1, cases=[0, 9])


def test_lexicase_zero_count_on_empty_pool_returns_empty():
    assert tools.sel_lexicase([], 0) == []
    assert tools.sel_epsilon_lexicase([], 0) == []


def test_lexicase_float_case_index_raises_index_error(multi_obj, make):
    population = [make(multi_obj, [0], (1.0, 2.0))]

    bad_cases: Any = [1.0]
    with pytest.raises(IndexError, match="case index"):
        tools.sel_lexicase(population, 1, cases=bad_cases)


def test_lexicase_unevaluated_fitness_random_draw(single_obj):
    """Unevaluated individuals (zero cases) draw from the pool without error."""
    ind = single_obj([0])
    population = [ind]

    tools.rng.seed(7)
    chosen = tools.sel_lexicase(population, 3)

    assert len(chosen) == 3
    assert all(item is ind for item in chosen)


def test_epsilon_lexicase_unevaluated_fitness_random_draw(single_obj):
    ind = single_obj([0])
    population = [ind]

    tools.rng.seed(7)
    chosen = tools.sel_epsilon_lexicase(population, 3)

    assert len(chosen) == 3
    assert all(item is ind for item in chosen)


def test_lexicase_empty_pool_with_positive_sel_count_raises_index_error():
    with pytest.raises(IndexError):
        tools.sel_lexicase([], 1)


def test_epsilon_lexicase_empty_pool_with_positive_sel_count_raises_index_error():
    with pytest.raises(IndexError):
        tools.sel_epsilon_lexicase([], 1)


def test_lexicase_nan_fitness_falls_back_to_pool(multi_obj, make):
    nan_ind = make(multi_obj, [0], (float("nan"), 0.0))
    other = make(multi_obj, [1], (1.0, 0.0))
    population = [nan_ind, other]

    tools.rng.seed(0)
    chosen = tools.sel_lexicase(population, 5)

    assert len(chosen) == 5
    assert all(ind in population for ind in chosen)
