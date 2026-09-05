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
import numpy
from deap_er import Fitness, creator, tools

FIT = "SORT_FIT"
IND = "SORT_IND"


def _create_types() -> None:
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])


def _delete_types() -> None:
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _ind(values: tuple[float, ...]):
    individual = creator.__dict__[IND]()
    individual.fitness.values = values
    return individual


def _front(values: list[tuple[float, ...]]):
    return [_ind(val) for val in values]


class TestSortNonDominated:
    def setup_method(self):
        _create_types()

    def teardown_method(self):
        _delete_types()

    def test_sel_count_zero(self):
        assert tools.sort_non_dominated(_front([(1.0, 2.0)]), 0) == []

    def test_empty_population_keeps_one_front(self):
        assert tools.sort_non_dominated([], 3) == [[]]
        assert tools.sel_nsga_2([], 3) == []
        refs = numpy.array([[1.0, 0.0], [0.0, 1.0]])
        assert tools.sel_nsga_3([], 3, refs) == []

    def test_two_fronts(self):
        pop = _front([(1.0, 4.0), (2.0, 2.0), (4.0, 1.0), (3.0, 3.0)])
        fronts = tools.sort_non_dominated(pop, len(pop))
        assert len(fronts) == 2
        first_vals = {ind.fitness.values for ind in fronts[0]}
        assert first_vals == {(1.0, 4.0), (2.0, 2.0), (4.0, 1.0)}
        assert [ind.fitness.values for ind in fronts[1]] == [(3.0, 3.0)]

    def test_duplicate_fitness_same_front(self):
        pop = _front([(1.0, 3.0), (1.0, 3.0), (3.0, 1.0)])
        fronts = tools.sort_non_dominated(pop, len(pop))
        assert len(fronts) == 1
        assert len(fronts[0]) == 3

    def test_sel_count_stops_after_enough(self):
        pop = _front([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
        fronts = tools.sort_non_dominated(pop, 1)
        assert len(fronts) == 1
        assert [ind.fitness.values for ind in fronts[0]] == [(1.0, 1.0)]
