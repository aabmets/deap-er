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

import pytest
from deap_er import tools


def test_symbolic_regression_benchmarks_return_finite_floats():
    point_2d: Any = [1.5, 2.5]
    point_3d: Any = [1.5, 2.0, 0.5]
    point_1d: Any = [1.0]
    cases = (
        (tools.bm_ripple, point_2d),
        (tools.bm_sin_cos, point_2d),
        (tools.bm_unwrapped_ball, point_2d),
        (tools.bm_kotanchek, point_2d),
        (tools.bm_salustowicz_1d, point_1d),
        (tools.bm_salustowicz_2d, point_2d),
        (tools.bm_rational_polynomial_1, point_3d),
        (tools.bm_rational_polynomial_2, point_2d),
    )
    for func, point in cases:
        value = func(point)
        assert isinstance(value, float)
        assert value == value
    assert tools.bm_unwrapped_ball(point_2d) == pytest.approx(
        10 / (5 + (1.5 - 3) ** 2 + (2.5 - 3) ** 2)
    )
