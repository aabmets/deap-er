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
import pytest
from deap_er import Fitness, creator, tools

FIT = "HV_FIT"
IND = "HV_IND"


def _setup() -> None:
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])


def _teardown() -> None:
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _ind(values: tuple[float, ...]):
    individual = creator.__dict__[IND]()
    individual.fitness.values = values
    return individual


class TestHypervolume:
    def test_1(self):
        front = numpy.array([(a, a) for a in numpy.arange(1, 0, -0.01)])
        assert tools.hypervolume(front, [2, 2]) == pytest.approx(3.9601000000000033)

    def test_2(self):
        front = numpy.array([(a, a) for a in numpy.arange(2, 0, -0.2)])
        assert tools.hypervolume(front, [3, 3]) == pytest.approx(7.839999999999998)

    def test_3(self):
        front = numpy.array([(a, a, a) for a in numpy.arange(3, 0, -0.03)])
        assert tools.hypervolume(front, [4, 5, 6]) == pytest.approx(117.7934729999985)

    def test_4(self):
        front = numpy.array([(a, a, a) for a in numpy.arange(4, 0, -0.4)])
        assert tools.hypervolume(front, [4, 5, 6]) == pytest.approx(92.73599999999996)

    def test_5(self):
        front = numpy.array([(a, a, a, a) for a in numpy.arange(5, 0, -0.567)])
        assert tools.hypervolume(front, [9, 2, 7, 4]) == pytest.approx(303.0190427996165)

    def test_empty(self):
        assert tools.hypervolume(numpy.array([])) == 0.0
        assert tools.hypervolume([]) == 0.0

    def test_population(self):
        _setup()
        try:
            result = tools.hypervolume([_ind((1.0, 4.0)), _ind((4.0, 1.0))], [5.0, 5.0])
        finally:
            _teardown()
        assert result == pytest.approx(7.0)

    def test_auto_reference(self):
        _setup()
        try:
            result = tools.hypervolume([_ind((1.0, 4.0)), _ind((4.0, 1.0))])
        finally:
            _teardown()
        assert result > 0.0

    def test_ndarray_individual_uses_fitness(self):
        creator.create_type(FIT, Fitness, weights=(-1.0, -1.0))
        creator.create_type(IND, numpy.ndarray, fitness=creator.__dict__[FIT])
        try:
            individual = creator.__dict__[IND]([0.0, 0.0])
            individual.fitness.values = (1.0, 4.0)
            as_one = tools.hypervolume(individual, [5.0, 5.0])
            as_pop = tools.hypervolume([individual], [5.0, 5.0])
        finally:
            _teardown()
        assert as_one == pytest.approx(4.0)
        assert as_pop == pytest.approx(4.0)


class TestLeastContrib:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            tools.least_contrib([])

    def test_least_index(self):
        _setup()
        try:
            pop = [
                _ind((5.0, 5.0)),
                _ind((4.0, 6.0)),
                _ind((2.0, 7.0)),
                _ind((7.0, 4.0)),
            ]
            idx = tools.least_contrib(pop, [10.0, 10.0])
        finally:
            _teardown()
        assert idx == 1
