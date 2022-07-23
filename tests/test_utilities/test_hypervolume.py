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
from deap_er.tools.hypervolume import HyperVolume
from deap_er.tools.hypervolume.node import Node
import numpy


# ====================================================================================== #
class TestHyperVolume:

    def test_1(self):
        front = [(a, a) for a in numpy.arange(1, 0, -0.01)]
        ref = numpy.array([2, 2])
        hv = HyperVolume(ref)
        result = hv.compute(front)
        assert result == 3.9601000000000033

    # -------------------------------------------------------------------------------------- #
    def test_2(self):
        front = [(a, a) for a in numpy.arange(2, 0, -0.2)]
        ref = numpy.array([3, 3])
        hv = HyperVolume(ref)
        result = hv.compute(front)
        assert result == 7.839999999999998

    # -------------------------------------------------------------------------------------- #
    def test_3(self):
        front = [(a, a, a) for a in numpy.arange(3, 0, -0.03)]
        ref = numpy.array([4, 5, 6])
        hv = HyperVolume(ref)
        result = hv.compute(front)
        assert result == 117.7934729999985

    # -------------------------------------------------------------------------------------- #
    def test_4(self):
        front = [(a, a, a) for a in numpy.arange(4, 0, -0.4)]
        ref = numpy.array([4, 5, 6])
        hv = HyperVolume(ref)
        result = hv.compute(front)
        assert result == 92.73599999999996

    # -------------------------------------------------------------------------------------- #
    def test_5(self):
        front = [(a, a, a, a) for a in numpy.arange(5, 0, -0.567)]
        ref = numpy.array([9, 2, 7, 4])
        hv = HyperVolume(ref)
        result = hv.compute(front)
        assert result == 303.0190427996165

    # -------------------------------------------------------------------------------------- #
    def test_6(self):
        front = [(a, a, a, a) for a in numpy.arange(10, 0, -0.5)]
        ref = numpy.array([1])
        hv = HyperVolume(ref)
        result = hv.compute(front)
        assert result == 0.5

    # -------------------------------------------------------------------------------------- #
    def test_7(self):
        front = []
        ref = []
        hv = HyperVolume(ref)
        result = hv.compute(front)
        assert result == 0.0


# ====================================================================================== #
class TestNode:

    def test_1(self):
        n1 = Node(1)
        n2 = Node(1)
        assert not n1 == n2
        assert not n1 != n2

    # -------------------------------------------------------------------------------------- #
    def test_2(self):
        n1 = Node(1, (1, 2, 3))
        n2 = Node(1, (1, 2, 3))
        assert n1 == n2
        assert n1 >= n2
        assert n1 <= n2

    # -------------------------------------------------------------------------------------- #
    def test_3(self):
        n1 = Node(1, (1, 2, 3))
        n2 = Node(1, (2, 3, 4))
        assert n1 != n2
        assert n1 < n2
        assert n2 > n1

    # -------------------------------------------------------------------------------------- #
    def test_4(self):
        n1 = Node(1, (1, 2, 3))
        n2 = Node(1, (9, 2, 3))
        assert n1 <= n2
        assert n2 >= n1

    # -------------------------------------------------------------------------------------- #
    def test_5(self):
        n = Node(1, (1, 2, 3, 4, 5))
        assert str(n) == '(1, 2, 3, 4, 5)'

    # -------------------------------------------------------------------------------------- #
    def test_6(self):
        data = (1, 2, 3)
        n = Node(1, data)
        assert hash(n) == hash(data)
