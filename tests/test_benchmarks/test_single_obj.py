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
import pytest
from deap_er import tools


def test_rand_is_in_unit_interval():
    tools.seed(0)
    ignored: Any = [1.0, 2.0]
    (value,) = tools.bm_rand(ignored)
    assert 0.0 <= value < 1.0


def test_plane_returns_the_first_gene():
    individual: Any = [3.5, 9.0]
    assert tools.bm_plane(individual) == (3.5,)


def test_sphere_is_zero_at_the_origin():
    origin: Any = [0.0, 0.0, 0.0]
    offset: Any = [1.0, 2.0]
    assert tools.bm_sphere(origin) == (0.0,)
    assert tools.bm_sphere(offset) == (5.0,)


def test_cigar_and_rosenbrock_known_optima():
    origin: Any = [0.0, 0.0, 0.0]
    cigar: Any = [1.0, 0.001]
    ones: Any = [1.0, 1.0, 1.0]
    zeros: Any = [0.0, 0.0]
    assert tools.bm_cigar(origin) == (0.0,)
    assert tools.bm_cigar(cigar) == pytest.approx((1.0 + 1e6 * 1e-6,))
    assert tools.bm_rosenbrock(ones) == (0.0,)
    assert tools.bm_rosenbrock(zeros)[0] > 0.0


def test_h1_and_ackley_at_documented_points():
    peak: Any = [8.6998, 6.7665]
    origin2: Any = [0.0, 0.0]
    origin3: Any = [0.0, 0.0, 0.0]
    offset: Any = [1.0, -1.0]
    (h1,) = tools.bm_h1(peak)
    assert h1 >= 0.0
    assert tools.bm_h1(origin2)[0] >= 0.0
    assert tools.bm_ackley(origin3) == pytest.approx((0.0,), abs=1e-12)
    assert tools.bm_ackley(offset)[0] > 0.0


def test_bohachevsky_griewank_schaffer_schwefel():
    origin: Any = [0.0, 0.0, 0.0]
    offset: Any = [1.0, -0.5, 0.25]
    assert tools.bm_bohachevsky(origin) == pytest.approx((0.0,), abs=1e-12)
    assert tools.bm_bohachevsky(offset)[0] > 0.0
    assert tools.bm_griewank(origin) == pytest.approx((0.0,), abs=1e-12)
    assert tools.bm_griewank(offset)[0] > 0.0
    assert tools.bm_schaffer(origin) == pytest.approx((0.0,), abs=1e-12)
    assert tools.bm_schaffer(offset)[0] > 0.0
    assert tools.bm_schwefel(offset)[0] > 0.0


def test_himmelblau_and_rastrigin_variants():
    himmel: Any = [3.0, 2.0]
    origin2: Any = [0.0, 0.0]
    origin3: Any = [0.0, 0.0, 0.0]
    offset2: Any = [1.0, -1.0]
    scaled: Any = [0.2, -0.3, 0.1]
    skewed: Any = [0.2, -0.3]
    assert tools.bm_himmelblau(himmel) == pytest.approx((0.0,), abs=1e-12)
    assert tools.bm_himmelblau(origin2)[0] > 0.0
    assert tools.bm_rastrigin(origin2) == pytest.approx((0.0,), abs=1e-12)
    assert tools.bm_rastrigin(offset2)[0] > 0.0
    assert tools.bm_rastrigin_scaled(origin3) == pytest.approx((0.0,), abs=1e-12)
    assert tools.bm_rastrigin_scaled(scaled)[0] > 0.0
    assert tools.bm_rastrigin_skewed(origin2) == pytest.approx((0.0,), abs=1e-12)
    assert tools.bm_rastrigin_skewed(skewed)[0] > 0.0


def test_shekel_sums_inverse_squared_distances():
    individual: Any = [0.0, 0.0]
    matrix = numpy.array([[0.0, 0.0], [1.0, 1.0]])
    vector = numpy.array([0.1, 0.2])
    (value,) = tools.bm_shekel(individual, matrix, vector)
    assert value == pytest.approx(1 / 0.1 + 1 / (0.2 + 2.0))
