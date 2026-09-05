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
