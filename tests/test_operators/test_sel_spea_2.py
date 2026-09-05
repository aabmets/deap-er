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
        (5, [7, 8, 5, 9, 4]),
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
