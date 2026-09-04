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
from deap_er import base, creator, tools

MO_FIT = "MET_FIT"
MO_IND = "MET_IND"


def _setup() -> None:
    creator.create(MO_FIT, base.Fitness, weights=(-1.0, -1.0))
    creator.create(MO_IND, list, fitness=creator.__dict__[MO_FIT])


def _teardown() -> None:
    del creator.__dict__[MO_FIT]
    del creator.__dict__[MO_IND]


def _ind(values: tuple[float, float]):
    individual = creator.__dict__[MO_IND](list(values))
    individual.fitness.values = values
    return individual


def test_nsga_diversity_for_one_and_many_points():
    _setup()
    try:
        first: Any = (0.0, 1.0)
        last: Any = (1.0, 0.0)
        alone = [_ind((0.5, 0.5))]
        front = [_ind((0.0, 1.0)), _ind((0.5, 0.5)), _ind((1.0, 0.0))]
        one = tools.nsga_diversity(alone, first, last)
        many = tools.nsga_diversity(front, first, last)
    finally:
        _teardown()
    assert one > 0.0
    assert many >= 0.0


def test_nsga_convergence_and_inverted_generational_distance():
    _setup()
    try:
        front = [_ind((0.1, 0.9)), _ind((0.9, 0.1))]
        optimal: Any = [(0.0, 1.0), (1.0, 0.0)]
        approx: Any = [(0.1, 0.9), (0.9, 0.1)]
        conv = tools.nsga_convergence(front, optimal)
        igd = tools.inv_gen_dist(optimal, approx)
    finally:
        _teardown()
    assert conv > 0.0
    assert igd == pytest.approx(0.1414213562373095, rel=1e-6)
