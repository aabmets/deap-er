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
from deap_er.creator import overrides
from deap_er import creator
import pytest
import numpy
import array


# -------------------------------------------------------------------------------------- #
def test_creator_create():
    creator.create('TestClass_0', int)
    assert hasattr(creator, 'TestClass_0')


# -------------------------------------------------------------------------------------- #
def test_creator_inheritance():
    creator.create('TestClass_1', int)
    assert isinstance(creator.TestClass_1(), int)


# -------------------------------------------------------------------------------------- #
def test_creator_warning():
    creator.create('TestClass_2', int)
    with pytest.warns(RuntimeWarning):
        creator.create('TestClass_2', int)


# -------------------------------------------------------------------------------------- #
def test_creator_custom_attr_class():
    creator.create('TestClass_3', int, my_attr=set())
    assert hasattr(creator.TestClass_3, 'my_attr')


# -------------------------------------------------------------------------------------- #
def test_creator_custom_attr_instance():
    creator.create('TestClass_4', int, my_attr=set)
    assert hasattr(creator.TestClass_4(), 'my_attr')


# -------------------------------------------------------------------------------------- #
def test_creator_numpy_override():
    creator.create('TestClass_5', numpy.ndarray)
    x = creator.TestClass_5([])
    y = overrides.NumpyOverride
    assert isinstance(x, y)


# -------------------------------------------------------------------------------------- #
def test_creator_array_override():
    creator.create('TestClass_6', array.array)
    x = creator.TestClass_6('I', [])
    y = overrides.ArrayOverride
    assert isinstance(x, y)


# -------------------------------------------------------------------------------------- #
def test_creator_base_class_instance():
    creator.create('TestClass_7', int())
    x = creator.TestClass_7()
    assert isinstance(x, int)
