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
from deap_er.creator import overrides
from deap_er.creator import creator
import pytest
import numpy
import array


CNAME = "CLASS_NAME"


# ====================================================================================== #
def test_creator_overwrite_warning():
    creator.create(CNAME, int)
    with pytest.warns(RuntimeWarning):
        creator.create(CNAME, int)
    creator.__dict__.pop(CNAME)


# ====================================================================================== #
class TestCreatorBasicFunctionality:

    # -------------------------------------------------------- #
    def test_creation(self):
        creator.create(CNAME, int)
        assert hasattr(creator, CNAME)
        assert issubclass(creator.__dict__[CNAME], int)
        creator.__dict__.pop(CNAME)

    # -------------------------------------------------------- #
    def test_class_attr(self):
        creator.create(CNAME, int, my_attr=int())
        assert hasattr(creator.__dict__[CNAME], 'my_attr')
        assert hasattr(creator.__dict__[CNAME](), 'my_attr')
        creator.__dict__.pop(CNAME)

    # -------------------------------------------------------- #
    def test_instance_attr(self):
        creator.create(CNAME, int, my_attr=int)
        assert not hasattr(creator.__dict__[CNAME], 'my_attr')
        assert hasattr(creator.__dict__[CNAME](), 'my_attr')
        creator.__dict__.pop(CNAME)

    # -------------------------------------------------------- #
    def test_list_creation(self):
        creator.create(CNAME, list)
        obj = creator.__dict__[CNAME]([1, 2, 3, 4])
        assert obj == [1, 2, 3, 4]
        creator.__dict__.pop(CNAME)

    # -------------------------------------------------------- #
    def test_list_attr(self):
        creator.create(CNAME, list, a=1)
        obj = creator.__dict__[CNAME]([1, 2, 3, 4])
        assert obj.a == 1
        creator.__dict__.pop(CNAME)


# ====================================================================================== #
class TestCreatorNumpy:
    data = [x for x in range(10)]

    # -------------------------------------------------------- #
    def test_ndarray_class_override(self):
        creator.create(CNAME, numpy.ndarray)
        a = creator.__dict__[CNAME]([])
        b = overrides._NumpyOverride
        assert isinstance(a, b)
        creator.__dict__.pop(CNAME)

    # -------------------------------------------------------- #
    def test_ndarray_instance_override(self):
        creator.create(CNAME, numpy.ndarray([]))
        a = creator.__dict__[CNAME]([])
        b = overrides._NumpyOverride
        assert isinstance(a, b)
        creator.__dict__.pop(CNAME)

    # -------------------------------------------------------- #
    def test_ndarray_values(self):
        creator.create(CNAME, numpy.ndarray([]))
        obj = creator.__dict__[CNAME](self.data)
        assert all(map(lambda x, y: x == y, obj, self.data))
        creator.__dict__.pop(CNAME)

    # -------------------------------------------------------- #
    def test_array_class_override(self):
        creator.create(CNAME, numpy.array)
        a = creator.__dict__[CNAME]([])
        b = overrides._NumpyOverride
        assert isinstance(a, b)
        creator.__dict__.pop(CNAME)

    # -------------------------------------------------------- #
    def test_array_instance_override(self):
        creator.create(CNAME, numpy.array([]))
        a = creator.__dict__[CNAME]([])
        b = overrides._NumpyOverride
        assert isinstance(a, b)
        creator.__dict__.pop(CNAME)

    # -------------------------------------------------------- #
    def test_array_values(self):
        creator.create(CNAME, numpy.array([]))
        obj = creator.__dict__[CNAME](self.data)
        assert all(map(lambda x, y: x == y, obj, self.data))
        creator.__dict__.pop(CNAME)

    # -------------------------------------------------------- #
    def test_various_4(self):
        creator.create(CNAME, numpy.ndarray)
        a = creator.__dict__[CNAME]([1, 2, 3, 4])
        b = creator.__dict__[CNAME]([5, 6, 7, 8])

        a[1:3], b[1:3] = b[1:3], a[1:3]
        ta = numpy.array([1, 6, 7, 4])
        tb = numpy.array([5, 6, 7, 8])
        assert all(a == ta)
        assert all(b == tb)
        creator.__dict__.pop(CNAME)

    # -------------------------------------------------------- #
    def test_various_5(self):
        creator.create(CNAME, numpy.ndarray)
        a = creator.__dict__[CNAME]([1, 2, 3, 4])
        b = creator.__dict__[CNAME]([5, 6, 7, 8])

        a[1:3], b[1:3] = b[1:3].copy(), a[1:3].copy()
        ta = numpy.array([1, 6, 7, 4])
        tb = numpy.array([5, 2, 3, 8])
        assert all(a == ta)
        assert all(b == tb)
        creator.__dict__.pop(CNAME)


# ====================================================================================== #
class TestCreatorBuiltinsArray:
    data = [x for x in range(10)]

    # -------------------------------------------------------- #
    def test_array_override(self):
        creator.create(CNAME, array.array, typecode='i')
        a = creator.__dict__[CNAME]([])
        b = overrides._ArrayOverride
        assert isinstance(a, b)
        creator.__dict__.pop(CNAME)

    # -------------------------------------------------------- #
    def test_array_values(self):
        creator.create(CNAME, array.array, typecode='i')
        obj = creator.__dict__[CNAME](self.data)
        assert all(map(lambda x, y: x == y, obj, self.data))
        creator.__dict__.pop(CNAME)

    def test_array_comparison(self):
        creator.create(CNAME, array.array, typecode="i")
        a = creator.__dict__[CNAME]([1, 2, 3, 4])
        b = creator.__dict__[CNAME]([5, 6, 7, 8])

        a[1:3], b[1:3] = b[1:3], a[1:3]
        ta = array.array("i", [1, 6, 7, 4])
        tb = array.array("i", [5, 2, 3, 8])
        assert a == ta
        assert b == tb
        creator.__dict__.pop(CNAME)
