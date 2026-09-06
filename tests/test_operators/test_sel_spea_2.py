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
from deap_er import Fitness, creator, tools
from deap_er.private.operators.sel_spea_2_helpers import fill_from_density, raw_fitness

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


def _spea2_population(multi_obj, make):
    return [make(multi_obj, [i], value) for i, value in enumerate(SPEA2_VALUES)]


@pytest.mark.parametrize(
    ("sel_count", "expected"),
    [
        (3, [0, 4, 8]),
        (5, [0, 1, 4, 8, 9]),
        (10, [0, 1, 2, 3, 4, 8, 9, 6, 5, 7]),
    ],
)
def test_spea2_selection_is_stable(multi_obj, make, sel_count, expected):
    # Characterization: sel_count below the first front size takes the archive
    # truncation path, above it takes the density path, which consumes RNG.
    population = _spea2_population(multi_obj, make)

    tools.rng.seed(2024)
    chosen = tools.sel_spea_2(population, sel_count)

    assert [ind[0] for ind in chosen] == expected


def test_spea2_returns_requested_count(multi_obj, make):
    population = _spea2_population(multi_obj, make)

    tools.rng.seed(11)
    assert len(tools.sel_spea_2(population, 4)) == 4


@pytest.mark.parametrize(
    ("sel_count", "expected"),
    [
        (2, [7, 8]),
        (5, [7, 8, 9, 5, 6]),
    ],
)
def test_spea2_mixed_sign_weights_change_the_archive(make, sel_count, expected):
    # Dominance uses wvalues, so maximizing the first objective and
    # minimizing the second is not the same front as (1, 1) weights.
    # Distances stay in objective-value space.
    creator.create_type("SEL_MS_FIT", Fitness, weights=(1.0, -1.0))
    creator.create_type("SEL_MS_IND", list, fitness=creator.__dict__["SEL_MS_FIT"])
    try:
        population = [
            make(creator.__dict__["SEL_MS_IND"], [i], value) for i, value in enumerate(SPEA2_VALUES)
        ]
        tools.rng.seed(2024)
        chosen = tools.sel_spea_2(population, sel_count)
        assert [ind[0] for ind in chosen] == expected
    finally:
        del creator.__dict__["SEL_MS_FIT"]
        del creator.__dict__["SEL_MS_IND"]


def test_spea2_density_includes_isolated_point(make):
    # (0,0) is the only non-dominated member. The last-indexed isolate
    # (10,10) must get a real k-th neighbour, not a zero-padded row.
    creator.create_type("SEL_S2D_FIT", Fitness, weights=(-1.0, -1.0))
    creator.create_type("SEL_S2D_IND", list, fitness=creator.__dict__["SEL_S2D_FIT"])
    try:
        pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (10.0, 10.0)]
        population = [
            make(creator.__dict__["SEL_S2D_IND"], [i], value) for i, value in enumerate(pts)
        ]
        fits = raw_fitness(population)
        isolated_raw = fits[3]
        tools.rng.seed(0)
        fill_from_density(population, [0], fits, 3)
        tools.rng.seed(0)
        chosen = tools.sel_spea_2(population, 3)
        assert 0 in [ind[0] for ind in chosen]
        # Zero-padded last row yields k-th distance 0 and density add 1/2.
        assert fits[3] - isolated_raw < 0.5
    finally:
        del creator.__dict__["SEL_S2D_FIT"]
        del creator.__dict__["SEL_S2D_IND"]


def test_spea2_density_fill_prefers_tied_isolate(make):
    # Same raw fitness on the second front: a tight cluster plus an
    # isolated point. Density fill to 2 must keep the isolate.
    creator.create_type("SEL_S2T_FIT", Fitness, weights=(-1.0, -1.0))
    creator.create_type("SEL_S2T_IND", list, fitness=creator.__dict__["SEL_S2T_FIT"])
    try:
        pts = [(0.0, 0.0), (1.0, 1.0), (1.05, 0.95), (1.1, 0.9), (0.1, 10.0)]
        population = [
            make(creator.__dict__["SEL_S2T_IND"], [i], value) for i, value in enumerate(pts)
        ]
        assert raw_fitness(population) == [0.0, 4.0, 4.0, 4.0, 4.0]
        tools.rng.seed(0)
        chosen = tools.sel_spea_2(population, 2)
        assert [ind[0] for ind in chosen] == [0, 4]
    finally:
        del creator.__dict__["SEL_S2T_FIT"]
        del creator.__dict__["SEL_S2T_IND"]


def test_spea2_empty_population_returns_empty():
    assert tools.sel_spea_2([], 1) == []
