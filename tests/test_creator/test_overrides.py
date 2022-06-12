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
from deap_er.creator import overrides as ovr
from copy import deepcopy
import pickle
import array
import numpy


# ====================================================================================== #
class TestNumpyOverrideClass:
    data = (x for x in range(0, 10))

    def test_numpy_override_instantiation(self):
        obj = ovr._NumpyOverride(self.data)
        assert isinstance(obj, ovr._NumpyOverride)
        assert issubclass(ovr._NumpyOverride, numpy.ndarray)

    # -------------------------------------------------------------------------------------- #
    def test_numpy_override_deepcopy(self):
        obj = ovr._NumpyOverride(self.data)
        copy = deepcopy(obj)
        assert isinstance(copy, numpy.ndarray)
        assert all(obj == copy)
        assert obj.__dict__ == copy.__dict__

    # -------------------------------------------------------------------------------------- #
    def test_numpy_override_pickling(self):
        obj = ovr._NumpyOverride(self.data)
        jar = pickle.dumps(obj)
        copy = pickle.loads(jar)
        assert all(obj == copy)
        assert obj.__dict__ == copy.__dict__

    # -------------------------------------------------------------------------------------- #
    def test_array_override_reduction(self):
        obj = ovr._NumpyOverride(self.data)
        cls, args, state = obj.__reduce__()
        assert cls == ovr._NumpyOverride
        assert isinstance(args, tuple)
        assert isinstance(state, dict)


# ====================================================================================== #
class TestArrayOverrideClass:
    data = (x for x in range(0, 10))

    def test_array_override_instantiation(self):
        obj = ovr._ArrayOverride('I', self.data)
        assert isinstance(obj, ovr._ArrayOverride)
        assert issubclass(ovr._ArrayOverride, array.array)

    # -------------------------------------------------------------------------------------- #
    def test_array_override_deepcopy(self):
        obj = ovr._ArrayOverride('I', self.data)
        copy = deepcopy(obj)
        assert isinstance(copy, array.array)
        assert obj == copy
        assert obj.__dict__ == copy.__dict__

    # -------------------------------------------------------------------------------------- #
    def test_array_override_pickling(self):
        obj = ovr._ArrayOverride('I', self.data)
        jar = pickle.dumps(obj)
        copy = pickle.loads(jar)
        assert obj == copy
        assert obj.__dict__ == copy.__dict__

    # -------------------------------------------------------------------------------------- #
    def test_array_override_reduction(self):
        obj = ovr._ArrayOverride('I', self.data)
        cls, args, state = obj.__reduce__()
        assert cls == ovr._ArrayOverride
        assert isinstance(args, tuple)
        assert isinstance(state, dict)
