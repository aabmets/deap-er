#
#   Apache License 2.0
#
#   Copyright (c) 2022, Mattias Aabmets
#
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#
#   SPDX-License-Identifier: Apache-2.0
#
from typing import Any

import numpy
from deap_er import tools
from deap_er.private.operators.bounds import broadcast_param


def test_broadcast_param_accepts_numpy_0d_float_bounds():
    scalar_low: Any = numpy.array(0.0)
    scalar_up: Any = numpy.asarray(1.0)
    low = broadcast_param("low", scalar_low, 3)
    up = broadcast_param("up", scalar_up, 3)
    assert list(low) == [0.0, 0.0, 0.0]
    assert list(up) == [1.0, 1.0, 1.0]


def test_broadcast_param_accepts_numpy_0d_integer_bounds():
    scalar_low: Any = numpy.array(0)
    scalar_up: Any = numpy.array(3, dtype=numpy.int64)
    low = broadcast_param("low", scalar_low, 2)
    up = broadcast_param("up", scalar_up, 2)
    assert list(low) == [0, 0]
    assert list(up) == [3, 3]


def test_bounded_operators_accept_numpy_0d_bounds():
    low: Any = numpy.array(0.0)
    up: Any = numpy.array(1.0)
    tools.rng.seed(2)
    individual: Any = [0.5, 0.5]
    (mutant,) = tools.mut_gaussian_bounded(individual, 0.0, 1.0, low, up, 1.0)
    assert all(0.0 <= gene <= 1.0 for gene in mutant)
    tools.rng.seed(3)
    poly_ind: Any = [0.5]
    (poly,) = tools.mut_polynomial_bounded(poly_ind, 20.0, low, up, 1.0)
    assert 0.0 <= poly[0] <= 1.0
    left: Any = [0.2, 0.3]
    right: Any = [0.8, 0.7]
    child1, child2 = tools.cx_blend_bounded(left, right, 0.5, low, up)
    assert all(0.0 <= gene <= 1.0 for gene in child1)
    assert all(0.0 <= gene <= 1.0 for gene in child2)


def test_mut_uniform_int_accepts_numpy_0d_integer_bounds():
    individual: Any = [0, 0, 0]
    low: Any = numpy.array(0)
    up: Any = numpy.array(3)
    tools.rng.seed(1)
    (mutant,) = tools.mut_uniform_int(individual, low, up, 1.0)
    assert all(0 <= gene <= 3 for gene in mutant)
