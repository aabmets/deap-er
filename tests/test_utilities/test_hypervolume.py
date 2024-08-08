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
from deap_er.utilities.hypervolume import HyperVolume
from deap_er.utilities.hypervolume.node import Node
import numpy


# ====================================================================================== #
class TestHyperVolume:

    def test_1(self):
        front = [(a, a) for a in numpy.arange(1, 0, -0.01)]
        front = numpy.array(front)
        ref = numpy.array([2, 2])
        hv = HyperVolume(ref)
        result = hv.compute(front)
        assert result == 3.9601000000000033

    # -------------------------------------------------------------------------------------- #
    def test_2(self):
        front = [(a, a) for a in numpy.arange(2, 0, -0.2)]
        front = numpy.array(front)
        ref = numpy.array([3, 3])
        hv = HyperVolume(ref)
        result = hv.compute(front)
        assert result == 7.839999999999998

    # -------------------------------------------------------------------------------------- #
    def test_3(self):
        front = [(a, a, a) for a in numpy.arange(3, 0, -0.03)]
        front = numpy.array(front)
        ref = numpy.array([4, 5, 6])
        hv = HyperVolume(ref)
        result = hv.compute(front)
        assert result == 117.7934729999985

    # -------------------------------------------------------------------------------------- #
    def test_4(self):
        front = [(a, a, a) for a in numpy.arange(4, 0, -0.4)]
        front = numpy.array(front)
        ref = numpy.array([4, 5, 6])
        hv = HyperVolume(ref)
        result = hv.compute(front)
        assert result == 92.73599999999996

    # -------------------------------------------------------------------------------------- #
    def test_5(self):
        front = [(a, a, a, a) for a in numpy.arange(5, 0, -0.567)]
        front = numpy.array(front)
        ref = numpy.array([9, 2, 7, 4])
        hv = HyperVolume(ref)
        result = hv.compute(front)
        assert result == 303.0190427996165

    # -------------------------------------------------------------------------------------- #
    def test_6(self):
        front = [(a, a, a, a) for a in numpy.arange(10, 0, -0.5)]
        front = numpy.array(front)
        ref = numpy.array([1])
        hv = HyperVolume(ref)
        result = hv.compute(front)
        assert result == 0.5

    # -------------------------------------------------------------------------------------- #
    def test_7(self):
        front = numpy.array([])
        ref = numpy.array([])
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
