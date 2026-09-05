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


def _finite(values) -> None:
    assert all(isinstance(value, float) for value in values)
    assert all(value == value and abs(value) != float("inf") for value in values)


def test_kursawe_schaffer_fonseca_poloni_dent():
    k_vec: Any = [0.0, 0.5, -0.5]
    s_vec: Any = [1.0]
    f_vec: Any = [0.0, 0.0, 0.0]
    p_vec: Any = [0.0, 0.0]
    d_vec: Any = [0.2, -0.3]
    kursawe = tools.bm_kursawe(k_vec)
    schaffer = tools.bm_schaffer_mo(s_vec)
    fonseca = tools.bm_fonseca(f_vec)
    poloni = tools.bm_poloni(p_vec)
    dent = tools.bm_dent(d_vec)

    assert schaffer == pytest.approx((1.0, 1.0))
    assert len(kursawe) == len(fonseca) == len(poloni) == len(dent) == 2
    _finite(kursawe)
    _finite(fonseca)
    _finite(poloni)
    _finite(dent)


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


def test_dtlz_family_returns_the_requested_objective_count():
    vector: Any = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for count in (2, 3):
        for func in (
            tools.bm_dtlz_1,
            tools.bm_dtlz_2,
            tools.bm_dtlz_3,
            tools.bm_dtlz_5,
            tools.bm_dtlz_6,
            tools.bm_dtlz_7,
        ):
            values = func(vector, count)
            assert len(values) == count
            _finite(values)
        values = tools.bm_dtlz_4(vector, count, alpha=100.0)
        assert len(values) == count
        _finite(values)


def test_dtlz5_f1_uses_only_the_angular_variables():
    vector: Any = [0.25] + [0.5] * 6
    values = tools.bm_dtlz_5(vector, 3)
    assert values[0] == pytest.approx(0.6532814824381883, rel=1e-6)
    assert sum(value * value for value in values) == pytest.approx(1.0, rel=1e-6)
