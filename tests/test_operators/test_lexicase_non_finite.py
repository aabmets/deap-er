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
import math
from functools import partial

import numpy
import pytest
from deap_er import tools
from deap_er.private.operators import lexicase_vectorized

SELECTORS = {
    "strict": tools.sel_lexicase,
    "epsilon_auto": tools.sel_epsilon_lexicase,
    "epsilon_semi": partial(tools.sel_epsilon_lexicase, mode="epsilon_semi"),
    "epsilon_dynamic": partial(tools.sel_epsilon_lexicase, mode="epsilon_dynamic"),
    "epsilon_fixed": partial(tools.sel_epsilon_lexicase, epsilon=0.5),
    "batch": partial(tools.sel_batch_epsilon_lexicase, batch_size=1),
}


def _population(make, ind_cls, values):
    return [make(ind_cls, [i], (value,)) for i, value in enumerate(values)]


@pytest.mark.parametrize("select", SELECTORS.values(), ids=SELECTORS.keys())
def test_one_nan_does_not_randomize_lexicase(single_obj, make, select):
    # OP-1: one NaN made the case bound NaN, so every pick turned uniform.
    population = _population(make, single_obj, [10.0] + [0.0] * 8 + [math.nan])

    tools.rng.seed(3)
    chosen = select(population, 200)

    assert all(ind is population[0] for ind in chosen)


@pytest.mark.parametrize("select", SELECTORS.values(), ids=SELECTORS.keys())
def test_mostly_infinite_case_does_not_randomize_lexicase(single_obj, make, select):
    # OP-1: more than half the column at -inf made the MAD NaN.
    population = _population(make, single_obj, [10.0, 0.0, 0.0, 0.0] + [-math.inf] * 6)

    tools.rng.seed(4)
    chosen = select(population, 200)

    assert all(ind is population[0] for ind in chosen)


@pytest.mark.parametrize("select", SELECTORS.values(), ids=SELECTORS.keys())
def test_non_finite_values_rank_worst(single_obj, make, select):
    # Non-finite values are treated as worst, even +inf on a maximized case.
    population = _population(make, single_obj, [math.inf, math.nan, 1.0, -math.inf])

    tools.rng.seed(5)
    chosen = select(population, 50)

    assert all(ind is population[2] for ind in chosen)


def test_static_epsilon_builds_each_pass_mask_once(multi_obj, make, monkeypatch):
    # OP-4: population-MAD, population-elite modes recomputed every case's
    # pass mask on every selection.
    calls = []
    original = lexicase_vectorized.apply_epsilon_filter

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(lexicase_vectorized, "apply_epsilon_filter", counting)
    rows = numpy.random.default_rng(0).random((30, 2))
    population = [make(multi_obj, [i], tuple(row)) for i, row in enumerate(rows)]

    tools.rng.seed(6)
    chosen = tools.sel_epsilon_lexicase(population, 100, mode="epsilon_static")

    assert len(chosen) == 100
    assert len(calls) <= 2
