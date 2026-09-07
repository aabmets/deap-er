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
from typing import Any, override

import numpy
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


def test_nsga_convergence_uses_fitness_when_optimal_are_individuals():
    _setup()
    try:
        front = [_ind((0.1, 0.9)), _ind((0.9, 0.1))]
        optimal = [_ind((0.0, 1.0)), _ind((1.0, 0.0))]
        # Genes equal the objectives in `_ind`; overwrite genes so a
        # genotype read would score the wrong front.
        for individual in (*front, *optimal):
            individual[:] = [999.0, 888.0]
        conv = tools.nsga_convergence(front, optimal)
    finally:
        _teardown()
    assert conv == pytest.approx(0.1414213562373095, rel=1e-6)


def test_nsga_diversity_uses_fitness_when_extremes_are_individuals():
    _setup()
    try:
        front = [_ind((0.0, 1.0)), _ind((0.5, 0.5)), _ind((1.0, 0.0))]
        first = _ind((0.0, 1.0))
        last = _ind((1.0, 0.0))
        first[:] = [999.0, 888.0]
        last[:] = [777.0, 666.0]
        delta = tools.nsga_diversity(front, first, last)
    finally:
        _teardown()
    assert delta == pytest.approx(0.0, abs=1e-12)


def test_duplicate_count_counts_twins():
    assert tools.duplicate_count([[1], [1], [2], [1]]) == 2
    assert tools.duplicate_count(["a", "bb", "a"], key=len) == 1


def test_duplicate_count_empty_population():
    assert tools.duplicate_count([]) == 0


def test_duplicate_count_single_individual():
    assert tools.duplicate_count(["only"]) == 0


def test_duplicate_count_all_unique_hashable():
    assert tools.duplicate_count([0, 1, 2, 3]) == 0


def test_duplicate_count_all_duplicates_hashable():
    assert tools.duplicate_count([7, 7, 7, 7]) == 3


def test_duplicate_count_mixed_hashable():
    assert tools.duplicate_count([1, 2, 1, 3, 2, 1]) == 3


def test_duplicate_count_unhashable_all_same():
    assert tools.duplicate_count([[1], [1], [1]]) == 2


def test_duplicate_count_unhashable_all_unique():
    assert tools.duplicate_count([[1], [2], [3]]) == 0


def test_duplicate_count_key_returns_unhashable_list():
    assert tools.duplicate_count(["a", "bb", "ccc"], key=list) == 0
    assert tools.duplicate_count(["aa", "b", "aa"], key=list) == 1


class _SortableUnhashable:
    """Unhashable key with total ordering (exercises the sort branch)."""

    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = value

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _SortableUnhashable) and self.value == other.value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _SortableUnhashable):
            return NotImplemented
        return self.value < other.value


class _UnsortableUnhashable:
    """Unhashable key with no ordering (exercises the list fallback)."""

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _UnsortableUnhashable)

    def __lt__(self, other: object) -> bool:
        raise TypeError("no ordering")


def test_duplicate_count_sortable_unhashable_keys_use_sort_path():
    population = [
        _SortableUnhashable(2),
        _SortableUnhashable(1),
        _SortableUnhashable(2),
    ]
    assert tools.duplicate_count(population) == 1


def test_duplicate_count_unsortable_unhashable_keys_use_list_fallback():
    population = [
        _UnsortableUnhashable(),
        _UnsortableUnhashable(),
        _UnsortableUnhashable(),
    ]
    assert tools.duplicate_count(population) == 2


def test_duplicate_count_ndarray_individuals_count_twins():
    fit_name = "DUP_NP_FIT"
    ind_name = "DUP_NP_IND"
    creator.create_type(fit_name, Fitness, weights=(1.0,))
    creator.create_type(ind_name, numpy.ndarray, fitness=creator.__dict__[fit_name])
    try:
        twins = [creator.__dict__[ind_name]([1, 2]), creator.__dict__[ind_name]([1, 2])]
        unique = [creator.__dict__[ind_name]([1, 2]), creator.__dict__[ind_name]([3, 4])]
        assert tools.duplicate_count(twins) == 1
        assert tools.duplicate_count(unique) == 0
        assert tools.duplicate_count(twins, key=tuple) == 1
    finally:
        del creator.__dict__[ind_name]
        del creator.__dict__[fit_name]
