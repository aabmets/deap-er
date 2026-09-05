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


def test_zdt_family_returns_two_objectives():
    vector: Any = [0.1, 0.2, 0.3, 0.4, 0.5]
    for func in (
        tools.bm_zdt_1,
        tools.bm_zdt_2,
        tools.bm_zdt_3,
        tools.bm_zdt_4,
        tools.bm_zdt_6,
    ):
        values = func(vector)
        assert len(values) == 2
        _finite(values)
