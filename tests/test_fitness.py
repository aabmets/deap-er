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
from copy import deepcopy

import numpy
import pytest
from deap_er import Fitness


class TestFitness:
    def test_instantiation(self, monkeypatch):
        with pytest.raises(TypeError):
            Fitness()
        monkeypatch.setattr(Fitness, "weights", [1, 2, 3])
        Fitness()

    def test_values_bad_length(self, monkeypatch):
        monkeypatch.setattr(Fitness, "weights", [1, 2, 3])
        with pytest.raises(TypeError):
            Fitness([1, 2, 3, 4])

    def test_values_access(self, monkeypatch):
        monkeypatch.setattr(Fitness, "weights", [1, 2, 3])

        ft = Fitness()
        assert ft.is_valid() is False
        assert ft.values == tuple()

        ft.values = [2, 2, 2]
        assert ft.is_valid() is True
        assert ft.values == (2, 2, 2)
        assert ft.wvalues == (2, 4, 6)

        del ft.values
        assert ft.is_valid() is False
        assert ft.wvalues == tuple()

    def test_domination(self, monkeypatch):
        monkeypatch.setattr(Fitness, "weights", [1, 1, 1])
        ft1 = Fitness([2, 2, 2])
        ft2 = Fitness([2, 2, 3])
        ft3 = Fitness([1, 2, 3])

        assert not ft1.dominates(ft2)
        assert not ft1.dominates(ft3)

        assert ft2.dominates(ft1)
        assert ft2.dominates(ft3)

        assert not ft3.dominates(ft1)
        assert not ft3.dominates(ft2)

        assert not ft1.dominates(ft1)
        assert not ft1.dominates(Fitness([2, 2, 2]))

    def test_domination_respects_objective_slice(self, monkeypatch):
        monkeypatch.setattr(Fitness, "weights", [1, 1, 1])
        better_first = Fitness([3, 1, 1])
        better_rest = Fitness([2, 9, 9])

        assert not better_first.dominates(better_rest)
        assert not better_rest.dominates(better_first)
        assert better_first.dominates(better_rest, slc=slice(0, 1))
        assert better_rest.dominates(better_first, slc=slice(1, None))

    def test_domination_uses_weighted_values(self, monkeypatch):
        # Maximize the first objective, minimize the second. Raw values make
        # the second look worse for `low_second`; wvalues reverse that.
        monkeypatch.setattr(Fitness, "weights", [1, -1])
        low_second = Fitness([2, 5])
        high_second = Fitness([1, 10])

        assert low_second.wvalues == (2.0, -5.0)
        assert high_second.wvalues == (1.0, -10.0)
        assert low_second.dominates(high_second)
        assert not high_second.dominates(low_second)

    @pytest.mark.parametrize("weights", [(1.0,), (1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)])
    def test_dominates_invalid_fitness_is_false(self, weights, monkeypatch):
        monkeypatch.setattr(Fitness, "weights", list(weights))
        valid = Fitness([1.0] * len(weights))
        invalid = Fitness()

        assert valid.dominates(invalid) is False
        assert invalid.dominates(valid) is False
        assert invalid.dominates(invalid) is False

    def test_comparison(self, monkeypatch):
        monkeypatch.setattr(Fitness, "weights", [1, 1, 1])
        ft1 = Fitness([2, 2, 2])
        ft2 = Fitness([3, 3, 3])
        ft3 = Fitness([4, 4, 4])

        assert ft3 > ft2
        assert ft3 >= ft3
        assert ft2 <= ft2
        assert ft1 < ft2
        assert ft1 == Fitness([2, 2, 2])
        assert ft1 != ft3

    def test_helper_methods(self, monkeypatch):
        monkeypatch.setattr(Fitness, "weights", [1, 1, 1])
        ft1 = Fitness([2, 2, 2])
        ft2 = Fitness([3, 3, 3])

        assert hash(ft1) != hash(ft2)
        assert ft1.__str__() == "(2.0, 2.0, 2.0)"
        assert ft1 == deepcopy(ft1)

    @pytest.mark.parametrize("zero", [0, 0.0])
    def test_zero_is_a_real_objective_value(self, zero, monkeypatch):
        monkeypatch.setattr(Fitness, "weights", [-1])

        ft = Fitness(zero)

        assert ft.is_valid() is True
        assert ft.values == (0.0,)

    def test_no_values_stays_invalid(self, monkeypatch):
        monkeypatch.setattr(Fitness, "weights", [-1])

        assert Fitness().is_valid() is False

    @pytest.mark.parametrize(
        "scalar",
        [numpy.float32(3.0), numpy.float64(3.0), numpy.int32(3), numpy.int64(3)],
    )
    def test_numpy_scalars_are_accepted(self, scalar, monkeypatch):
        monkeypatch.setattr(Fitness, "weights", [1])

        ft = Fitness()
        ft.values = scalar

        assert ft.values == (3.0,)

    def test_numpy_array_of_values_is_accepted(self, monkeypatch):
        monkeypatch.setattr(Fitness, "weights", [1, 1, 1])

        ft = Fitness()
        ft.values = numpy.array([1.0, 2.0, 3.0])

        assert ft.values == (1.0, 2.0, 3.0)

    def test_dominates_four_objectives_and_length(self, monkeypatch):
        monkeypatch.setattr(Fitness, "weights", [1, 1, 1, 1])
        better = Fitness([2, 2, 2, 3])
        worse = Fitness([2, 2, 2, 2])
        mixed = Fitness([3, 1, 2, 2])

        assert better.dominates(worse)
        assert not worse.dominates(better)
        assert not mixed.dominates(worse)
        assert not better.dominates(mixed)
        assert len(better) == 4

    def test_equality_with_non_fitness_returns_not_implemented(self, monkeypatch):
        monkeypatch.setattr(Fitness, "weights", [1])
        fitness = Fitness([1.0])

        assert fitness.__eq__(1.0) is NotImplemented
        assert fitness.__ne__(1.0) is NotImplemented
