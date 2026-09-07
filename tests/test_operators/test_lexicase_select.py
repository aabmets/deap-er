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
from deap_er.private.operators.sel_lexicase import lexicase_select
from deap_er.private.operators.sel_lexicase_matrix import lexicase_select_vectorized


def _keep(candidates, case, maximize):
    key = (
        (lambda ind: ind.fitness.values[case])
        if maximize
        else (lambda ind: -ind.fitness.values[case])
    )
    best = max(key(ind) for ind in candidates)
    return [ind for ind in candidates if key(ind) == best]


def test_lexicase_select_zero_count_returns_empty(multi_obj, make):
    population = [make(multi_obj, [0], (1.0, 2.0))]

    assert lexicase_select(population, 0, _keep) == []
    assert lexicase_select(population, -2, _keep) == []


def test_lexicase_select_filters_cases_and_breaks_ties(multi_obj, make):
    first = make(multi_obj, [0], (10.0, 0.0))
    second = make(multi_obj, [1], (0.0, 10.0))
    population = [first, second]

    chosen = lexicase_select(population, 12, _keep, cases=[0])

    assert len(chosen) == 12
    assert all(ind is first for ind in chosen)


def test_lexicase_select_empty_keep_falls_back_to_pool(multi_obj, make):
    population = [make(multi_obj, [0], (1.0, 2.0)), make(multi_obj, [1], (3.0, 4.0))]

    def drop_all(_candidates, _case, _maximize):
        return []

    chosen = lexicase_select(population, 6, drop_all)

    assert len(chosen) == 6
    assert all(ind in population for ind in chosen)


def test_lexicase_select_vectorized_zero_count(multi_obj, make):
    population = [make(multi_obj, [0], (1.0, 2.0))]

    chosen = lexicase_select_vectorized(
        population,
        0,
        __import__("numpy").array([[1.0, 2.0]]),
        [0, 1],
        (1.0, 1.0),
    )

    assert chosen == []
