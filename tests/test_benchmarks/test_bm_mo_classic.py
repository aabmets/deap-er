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
