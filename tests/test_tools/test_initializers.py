# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 - Mattias Aabmets, The DEAP Team and Other Contributors           #
#                                                                                        #
#   Permission is hereby granted, free of charge, to any person obtaining a copy         #
#   of this software and associated documentation files (the "Software"), to deal        #
#   in the Software without restriction, including without limitation the rights         #
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell            #
#   copies of the Software, and to permit persons to whom the Software is                #
#   furnished to do so, subject to the following conditions:                             #
#                                                                                        #
#   The above copyright notice and this permission notice shall be included in all       #
#   copies or substantial portions of the Software.                                      #
#                                                                                        #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR           #
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,             #
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE          #
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER               #
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,        #
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE        #
#   SOFTWARE.                                                                            #
#                                                                                        #
# ====================================================================================== #
from deap_er.tools import initializers as init


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
