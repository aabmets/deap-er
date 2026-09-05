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
from deap_er import Fitness, creator, tools

MO_FIT = "MET_FIT"
MO_IND = "MET_IND"


def _setup() -> None:
    creator.create_type(MO_FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(MO_IND, list, fitness=creator.__dict__[MO_FIT])


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


def test_nsga_diversity_is_invariant_to_front_order():
    _setup()
    try:
        first: Any = (0.0, 1.0)
        last: Any = (1.0, 0.0)
        ordered = [_ind((0.0, 1.0)), _ind((0.5, 0.5)), _ind((1.0, 0.0))]
        permuted = [_ind((0.5, 0.5)), _ind((1.0, 0.0)), _ind((0.0, 1.0))]
        sorted_delta = tools.nsga_diversity(ordered, first, last)
        shuffled_delta = tools.nsga_diversity(permuted, first, last)
    finally:
        _teardown()
    assert sorted_delta == pytest.approx(0.0, abs=1e-12)
    assert shuffled_delta == pytest.approx(0.0, abs=1e-12)


def test_nsga_diversity_for_a_single_point_is_one():
    _setup()
    try:
        first: Any = (0.0, 1.0)
        last: Any = (1.0, 0.0)
        alone = [_ind((0.5, 0.5))]
        delta = tools.nsga_diversity(alone, first, last)
    finally:
        _teardown()
    assert delta == pytest.approx(1.0)


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
    # Each front point is sqrt(0.02) from its nearest true-front vertex.
    assert conv == pytest.approx(0.1414213562373095, rel=1e-6)
    assert igd == pytest.approx(0.1414213562373095, rel=1e-6)


def test_duplicate_count_counts_twins():
    assert tools.duplicate_count([[1], [1], [2], [1]]) == 2
    assert tools.duplicate_count(["a", "bb", "a"], key=len) == 1
