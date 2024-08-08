#
#   MIT License
#   
#   Copyright (c) 2022, Mattias Aabmets
#   
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#   
#   SPDX-License-Identifier: MIT
#
from deap_er.base.fitness import Fitness
from copy import deepcopy
import pytest


# ====================================================================================== #
class TestFitness:

    def test_instantiation(self):
        with pytest.raises(TypeError):
            Fitness()
        Fitness.weights = [1, 2, 3]
        Fitness()

    # -------------------------------------------------------------------------------------- #
    def test_values_bad_length(self):
        Fitness.weights = [1, 2, 3]
        with pytest.raises(TypeError):
            Fitness([1, 2, 3, 4])

    # -------------------------------------------------------------------------------------- #
    def test_values_access(self):
        Fitness.weights = [1, 2, 3]

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

    # -------------------------------------------------------------------------------------- #
    def test_domination(self):
        Fitness.weights = [1, 1, 1]
        ft1 = Fitness([2, 2, 2])
        ft2 = Fitness([2, 2, 3])
        ft3 = Fitness([1, 2, 3])

        assert not ft1.dominates(ft2)
        assert not ft1.dominates(ft3)

        assert ft2.dominates(ft1)
        assert ft2.dominates(ft3)

        assert not ft3.dominates(ft1)
        assert not ft3.dominates(ft2)

    # -------------------------------------------------------------------------------------- #
    def test_comparison(self):
        Fitness.weights = [1, 1, 1]
        ft1 = Fitness([2, 2, 2])
        ft2 = Fitness([3, 3, 3])
        ft3 = Fitness([4, 4, 4])

        assert ft3 > ft2
        assert ft3 >= ft3
        assert ft2 <= ft2
        assert ft1 < ft2
        assert ft1 == ft1
        assert ft1 != ft3

    # -------------------------------------------------------------------------------------- #
    def test_helper_methods(self):
        Fitness.weights = [1, 1, 1]
        ft1 = Fitness([2, 2, 2])
        ft2 = Fitness([3, 3, 3])

        assert hash(ft1) != hash(ft2)
        assert ft1.__str__() == '(2.0, 2.0, 2.0)'
        assert ft1 == deepcopy(ft1)
