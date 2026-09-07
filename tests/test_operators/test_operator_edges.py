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
from deap_er.private.operators.sel_age_moea_2_curvature import estimate_curvature_nr


def test_polynomial_mutation_rejects_non_positive_eta():
    with pytest.raises(ValueError, match="eta"):
        individual: Any = [0.5]
        tools.mut_polynomial_bounded(individual, eta=0.0, low=0.0, up=1.0, mut_prob=1.0)


def test_gaussian_mutation_rejects_short_parameter_sequence():
    with pytest.raises(ValueError, match="at least the size"):
        individual: Any = [0.0, 1.0]
        tools.mut_gaussian(individual, mu=[0.0], sigma=0.1, mut_prob=1.0)


def test_curvature_rejects_wrong_shape_and_non_positive_points():
    assert estimate_curvature_nr(numpy.array([0.5, 0.5, 0.5]), 2) == 1.0
    assert estimate_curvature_nr(numpy.array([-1.0, -2.0]), 2) == 1.0


def test_curvature_overflow_returns_default():
    assert estimate_curvature_nr(numpy.array([1e200, 1e200]), 2) == 1.0


def test_assign_crowding_dist_empty_is_noop():
    tools.assign_crowding_dist([])


def test_uniform_reference_points_with_scaling():
    points = tools.uniform_reference_points(3, 4, scaling=0.5)
    assert points.shape[1] == 3
    assert numpy.all(points >= 0.0)
