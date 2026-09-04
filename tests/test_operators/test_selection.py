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
import random

import numpy
import pytest
from deap_er import base, creator, tools

SO_FIT = "SEL_SO_FIT"
SO_IND = "SEL_SO_IND"
MO_FIT = "SEL_MO_FIT"
MO_IND = "SEL_MO_IND"


@pytest.fixture
def single_obj():
    creator.create(SO_FIT, base.Fitness, weights=(1.0,))
    creator.create(SO_IND, list, fitness=creator.__dict__[SO_FIT])
    yield creator.__dict__[SO_IND]
    del creator.__dict__[SO_FIT]
    del creator.__dict__[SO_IND]


@pytest.fixture
def multi_obj():
    creator.create(MO_FIT, base.Fitness, weights=(1.0, 1.0))
    creator.create(MO_IND, list, fitness=creator.__dict__[MO_FIT])
    yield creator.__dict__[MO_IND]
    del creator.__dict__[MO_FIT]
    del creator.__dict__[MO_IND]


def _make(ind_cls, genes, values):
    ind = ind_cls(genes)
    ind.fitness.values = values
    return ind


@pytest.mark.parametrize("fitness_first", [True, False])
def test_double_tournament_returns_requested_count(single_obj, fitness_first):
    population = [_make(single_obj, [0] * (i % 4 + 1), (float(i),)) for i in range(12)]

    chosen = tools.sel_double_tournament(
        population,
        rounds=5,
        fitness_size=2,
        parsimony_size=1.4,
        fitness_first=fitness_first,
    )

    assert len(chosen) == 5
    assert all(ind in population for ind in chosen)


def test_double_tournament_favours_smaller_individuals(single_obj):
    # Equal fitness everywhere, so only the size tournament can decide.
    population = [_make(single_obj, [0] * length, (1.0,)) for length in (1, 1, 1, 20, 20, 20)]

    random.seed(4321)
    chosen = tools.sel_double_tournament(
        population,
        rounds=40,
        fitness_size=2,
        parsimony_size=2,
        fitness_first=True,
    )

    # parsimony_size=2 makes the shorter contestant win every non-tied contest.
    assert sum(len(ind) == 20 for ind in chosen) < sum(len(ind) == 1 for ind in chosen)


def test_stochastic_universal_sampling_zero_count(single_obj):
    population = [_make(single_obj, [i], (float(i + 1),)) for i in range(6)]

    assert tools.sel_stochastic_universal_sampling(population, 0) == []


def test_stochastic_universal_sampling_returns_requested_count(single_obj):
    population = [_make(single_obj, [i], (float(i + 1),)) for i in range(6)]

    chosen = tools.sel_stochastic_universal_sampling(population, 4)

    assert len(chosen) == 4
    assert all(ind in population for ind in chosen)


def test_epsilon_lexicase_zero_epsilon_is_strict(single_obj):
    # An explicit epsilon of 0.0 must not be treated as "compute it from the MAD".
    best = _make(single_obj, [0], (10.0,))
    population = [best] + [_make(single_obj, [i], (v,)) for i, v in enumerate((9.0, 8.0, 0.0), 1)]

    chosen = tools.sel_epsilon_lexicase(population, 20, epsilon=0.0)

    assert all(ind is best for ind in chosen)


def test_epsilon_lexicase_recomputes_epsilon_for_each_selection(multi_obj):
    # The two cases have very different spreads, so reusing the epsilon of one
    # case while filtering on the other changes which candidates survive.
    values = [(10.0, 0.0), (9.0, 50.0), (8.0, 100.0), (7.0, 150.0)]
    population = [_make(multi_obj, [i], value) for i, value in enumerate(values)]

    random.seed(99)
    batched = tools.sel_epsilon_lexicase(population, 8)
    random.seed(99)
    one_at_a_time = [tools.sel_epsilon_lexicase(population, 1)[0] for _ in range(8)]

    # Each selection must start from a fresh epsilon, so a batch of eight has to
    # match eight independent single selections drawn from the same seed.
    assert batched == one_at_a_time


SPEA2_VALUES = [
    (1.0, 9.0),
    (2.0, 8.0),
    (3.0, 7.0),
    (4.0, 6.0),
    (5.0, 5.0),
    (1.0, 1.0),
    (2.0, 2.0),
    (0.5, 0.5),
    (9.0, 1.0),
    (8.0, 2.0),
]


def _spea2_population(multi_obj):
    return [_make(multi_obj, [i], value) for i, value in enumerate(SPEA2_VALUES)]


@pytest.mark.parametrize(
    ("sel_count", "expected"),
    [
        (3, [0, 4, 8]),
        (5, [0, 1, 4, 8, 9]),
        (10, [0, 1, 2, 3, 4, 8, 9, 6, 5, 7]),
    ],
)
def test_spea2_selection_is_stable(multi_obj, sel_count, expected):
    # Characterization: sel_count below the first front size takes the archive
    # truncation path, above it takes the density path, which consumes RNG.
    population = _spea2_population(multi_obj)

    random.seed(2024)
    chosen = tools.sel_spea_2(population, sel_count)

    assert [ind[0] for ind in chosen] == expected


def test_spea2_returns_requested_count(multi_obj):
    population = _spea2_population(multi_obj)

    random.seed(11)
    assert len(tools.sel_spea_2(population, 4)) == 4


def test_nsga3_with_memory_updates_reference_points(multi_obj):
    ref_points = tools.uniform_reference_points(2, 4)
    select = tools.SelNSGA3WithMemory(ref_points)

    random.seed(7)
    for _ in range(2):
        population = [
            _make(multi_obj, [random.random()], (random.random(), random.random()))
            for _ in range(12)
        ]
        select(population, 6)

    assert numpy.all(numpy.isfinite(select.best_point))
    assert numpy.all(numpy.isfinite(select.worst_point))
    assert select.extreme_points is not None
