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


# ====================================================================================== #
def test_creator_overwrite_warning():
    creator.create('_DevTestClass', int)
    with pytest.warns(RuntimeWarning):
        creator.create('_DevTestClass', int)


# ====================================================================================== #
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
class TestCreatorBasicFunctionality:

    def test_create(self):
        creator.create('_DevTestClass', int)
        assert hasattr(creator, '_DevTestClass')

    # -------------------------------------------------------------------------------------- #
    def test_inheritance(self):
        creator.create('_DevTestClass', int)
        assert isinstance(creator._DevTestClass(), int)

    # -------------------------------------------------------------------------------------- #
    def test_custom_class_attr(self):
        creator.create('_DevTestClass', int, my_attr=int())
        assert hasattr(creator._DevTestClass, 'my_attr')
        assert hasattr(creator._DevTestClass(), 'my_attr')

    # -------------------------------------------------------------------------------------- #
    def test_custom_instance_attr(self):
        creator.create('_DevTestClass', int, my_attr=int)
        assert not hasattr(creator._DevTestClass, 'my_attr')
        assert hasattr(creator._DevTestClass(), 'my_attr')


# ====================================================================================== #
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
class TestCreatorNumpyNdarray:
    data = (x for x in range(10))

    def test_numpy_ndarray_class_override(self):
        creator.create('_DevTestClass', numpy.ndarray)
        a = creator._DevTestClass([])
        b = overrides._NumpyOverride
        assert isinstance(a, b)

    # -------------------------------------------------------------------------------------- #
    def test_numpy_ndarray_instance_override(self):
        creator.create('_DevTestClass', numpy.ndarray([]))
        a = creator._DevTestClass([])
        b = overrides._NumpyOverride
        assert isinstance(a, b)

    # -------------------------------------------------------------------------------------- #
    def test_numpy_ndarray_values(self):
        creator.create('_DevTestClass', numpy.ndarray([]))
        a = creator._DevTestClass(self.data)
        assert all(a == b for b in self.data)


# ====================================================================================== #
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
class TestCreatorNumpyArray:
    data = (x for x in range(10))

    def test_numpy_array_class_override(self):
        creator.create('_DevTestClass', numpy.array)
        a = creator._DevTestClass([])
        b = overrides._NumpyOverride
        assert isinstance(a, b)

    # -------------------------------------------------------------------------------------- #
    def test_numpy_array_instance_override(self):
        creator.create('_DevTestClass', numpy.array([]))
        a = creator._DevTestClass([])
        b = overrides._NumpyOverride
        assert isinstance(a, b)

    # -------------------------------------------------------------------------------------- #
    def test_numpy_array_values(self):
        creator.create('_DevTestClass', numpy.array([]))
        a = creator._DevTestClass(self.data)
        assert all(a == b for b in self.data)


# ====================================================================================== #
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
class TestCreatorBuiltinsArray:
    data = (x for x in range(10))

    def test_builtins_array_class_override(self):
        creator.create('_DevTestClass', array.array)
        a = creator._DevTestClass('I', [])
        b = overrides._ArrayOverride
        assert isinstance(a, b)

    # -------------------------------------------------------------------------------------- #
    def test_builtins_array_instance_override(self):
        creator.create('_DevTestClass', array.array('I', []))
        a = creator._DevTestClass('I', [])
        b = overrides._ArrayOverride
        assert isinstance(a, b)

    # -------------------------------------------------------------------------------------- #
    def test_builtins_array_values(self):
        creator.create('_DevTestClass', array.array('I', []))
        a = creator._DevTestClass('I', self.data)
        assert all(a == b for b in self.data)
