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
from typing import override

import numpy
from deap_er import Fitness, creator, tools


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
