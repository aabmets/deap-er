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
from deap_er.utilities import initializers as init


# ====================================================================================== #
def test_func_a() -> str:
    return 'gene'


def test_func_b() -> list:
    return [i for i in range(3)]


# ====================================================================================== #
class TestHelpers:

    def test_init_repeat_1(self):
        rtype = list
        count = 3
        result = init.init_repeat(rtype, test_func_a, count)
        assert isinstance(result, rtype)
        assert result.count('gene') == count
        assert len(result) == count

    # -------------------------------------------------------------------------------------- #
    def test_init_repeat_2(self):
        rtype = tuple
        count = 3
        result = init.init_repeat(rtype, test_func_a, count)
        assert isinstance(result, rtype)
        assert len(result) == count
        assert result.count('gene') == count

    # -------------------------------------------------------------------------------------- #
    def test_init_iterate_1(self):
        rtype = list
        result = init.init_iterate(rtype, test_func_b)
        assert isinstance(result, rtype)
        assert result == [0, 1, 2]

    # -------------------------------------------------------------------------------------- #
    def test_init_iterate_2(self):
        rtype = tuple
        result = init.init_iterate(rtype, test_func_b)
        assert isinstance(result, rtype)
        assert result == (0, 1, 2)

    # -------------------------------------------------------------------------------------- #
    def test_init_cycle_1(self):
        rtype = list
        count = 3
        funcs = {test_func_a, test_func_b}
        result = init.init_cycle(rtype, funcs, count)
        assert isinstance(result, rtype)
        assert len(result) == 6
        assert result.count('gene') == 3
        assert result.count([0, 1, 2]) == 3

    # -------------------------------------------------------------------------------------- #
    def test_init_cycle_2(self):
        rtype = tuple
        count = 3
        funcs = {test_func_a, test_func_b}
        result = init.init_cycle(rtype, funcs, count)
        assert isinstance(result, rtype)
        assert len(result) == 6
        assert result.count('gene') == 3
        assert result.count([0, 1, 2]) == 3
