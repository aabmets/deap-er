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
from deap_er import tools


def test_sel_tournament_returns_requested_count(single_obj, make):
    population = [make(single_obj, [i], (float(i),)) for i in range(10)]
    tools.rng.seed(2)
    chosen = tools.sel_tournament(population, rounds=7, contestants=3)
    assert len(chosen) == 7
    assert all(ind in population for ind in chosen)


def test_sel_tournament_empty_pool_raises():
    with pytest.raises(IndexError, match="empty"):
        tools.sel_tournament([], rounds=1, contestants=3)
    assert tools.sel_tournament([], rounds=0, contestants=3) == []


def test_sel_tournament_uses_fit_attr(single_obj, make):
    weak = make(single_obj, [0], (0.0,))
    strong = make(single_obj, [1], (100.0,))
    weak.alt = 100.0
    strong.alt = 0.0
    tools.rng.seed(1)
    via_fitness = tools.sel_tournament([weak, strong], rounds=40, contestants=2)
    tools.rng.seed(1)
    via_alt = tools.sel_tournament([weak, strong], rounds=40, contestants=2, fit_attr="alt")
    assert via_fitness.count(strong) > via_fitness.count(weak)
    assert via_alt.count(weak) > via_alt.count(strong)


@pytest.mark.parametrize("fitness_first", [True, False])
def test_double_tournament_returns_requested_count(single_obj, make, fitness_first):
    population = [make(single_obj, [0] * (i % 4 + 1), (float(i),)) for i in range(12)]

    chosen = tools.sel_double_tournament(
        population,
        rounds=5,
        fitness_size=2,
        parsimony_size=1.4,
        fitness_first=fitness_first,
    )

    assert len(chosen) == 5
    assert all(ind in population for ind in chosen)


def test_double_tournament_favours_smaller_individuals(single_obj, make):
    # Equal fitness everywhere, so only the size tournament can decide.
    population = [make(single_obj, [0] * length, (1.0,)) for length in (1, 1, 1, 20, 20, 20)]

    tools.rng.seed(4321)
    chosen = tools.sel_double_tournament(
        population,
        rounds=40,
        fitness_size=2,
        parsimony_size=2,
        fitness_first=True,
    )

    # parsimony_size=2 makes the shorter contestant win every non-tied contest.
    assert sum(len(ind) == 20 for ind in chosen) < sum(len(ind) == 1 for ind in chosen)


def test_sel_tournament_single_and_large_contestants(single_obj, make):
    population = [make(single_obj, [i], (float(i),)) for i in range(8)]
    tools.rng.seed(8)
    singles = tools.sel_tournament(population, rounds=5, contestants=1)
    tools.rng.seed(9)
    large = tools.sel_tournament(population, rounds=4, contestants=5)

    assert len(singles) == 5
    assert all(ind in population for ind in singles)
    assert len(large) == 4
    assert all(ind in population for ind in large)


def test_sel_tournament_rejects_non_positive_contestants(single_obj, make):
    population = [make(single_obj, [0], (1.0,))]
    with pytest.raises(ValueError, match="at least 1"):
        tools.sel_tournament(population, rounds=1, contestants=0)


def test_double_tournament_empty_pool_returns_empty():
    assert (
        tools.sel_double_tournament(
            [], rounds=5, fitness_size=2, parsimony_size=1.5, fitness_first=True
        )
        == []
    )
    assert (
        tools.sel_double_tournament(
            [], rounds=5, fitness_size=2, parsimony_size=1.5, fitness_first=False
        )
        == []
    )


def test_double_tournament_rejects_parsimony_outside_range(single_obj, make):
    population = [make(single_obj, [0], (1.0,))]
    with pytest.raises(ValueError, match="Parsimony"):
        tools.sel_double_tournament(
            population, rounds=1, fitness_size=2, parsimony_size=0.5, fitness_first=True
        )


def test_tournament_dcd_returns_exact_count_and_rejects_oversize(multi_obj, make):
    population = [make(multi_obj, [i], (float(i), float(10 - i))) for i in range(10)]
    tools.assign_crowding_dist(population)

    tools.rng.seed(3)
    chosen = tools.sel_tournament_dcd(population, 4)
    assert len(chosen) == 4

    tools.rng.seed(4)
    assert len(tools.sel_tournament_dcd(population, 1)) == 1
    assert len(tools.sel_tournament_dcd(population, 9)) == 9

    with pytest.raises(ValueError, match="less than or equal"):
        tools.sel_tournament_dcd(population, 11)
