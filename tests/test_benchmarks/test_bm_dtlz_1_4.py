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

from deap_er import tools


def _finite(values) -> None:
    assert all(isinstance(value, float) for value in values)
    assert all(value == value and abs(value) != float("inf") for value in values)


def test_dtlz_1_4_return_the_requested_objective_count():
    vector: Any = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for count in (2, 3):
        for func in (
            tools.bm_dtlz_1,
            tools.bm_dtlz_2,
            tools.bm_dtlz_3,
        ):
            values = func(vector, count)
            assert len(values) == count
            _finite(values)
        values = tools.bm_dtlz_4(vector, count, alpha=100.0)
        assert len(values) == count
        _finite(values)
