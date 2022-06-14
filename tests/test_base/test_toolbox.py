# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 - Mattias Aabmets and Contributors                                #
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
from deap_er.base.toolbox import Toolbox
from functools import partial
from copy import deepcopy


# ====================================================================================== #
class TestToolbox:

    def test_clone_func(self):
        tb = Toolbox()
        assert isinstance(tb.clone, partial)
        assert tb.clone.func == deepcopy

    # -------------------------------------------------------------------------------------- #
    def test_map_func(self):
        tb = Toolbox()
        assert isinstance(tb.clone, partial)
        assert tb.map.func == map

    # -------------------------------------------------------------------------------------- #
    def test_registration(self):
        tb = Toolbox()
        tb.register('__test__', str, 1)
        assert hasattr(tb, '__test__')
        tb.unregister('__test__')
        assert not hasattr(tb, '__test__')

    # -------------------------------------------------------------------------------------- #
    def test_execution(self):
        tb = Toolbox()
        tb.register('__test__', str, 1)
        assert tb.__test__() == '1'

    # -------------------------------------------------------------------------------------- #
    def test_decorator(self):
        def test_deco(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return result * 3
            return wrapper

        tb = Toolbox()
        tb.register('__test__', str, 1)
        tb.decorate('__test__', test_deco)
        assert tb.__test__() == '111'
