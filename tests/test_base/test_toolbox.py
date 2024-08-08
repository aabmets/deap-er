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
