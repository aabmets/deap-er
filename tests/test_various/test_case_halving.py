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

import pytest
from deap_er import Fitness, creator, tools

FIT = "HALV_FIT"
IND = "HALV_IND"


def _make_types(n_cases: int):
    creator.create_type(FIT, Fitness, weights=(-1.0,) * n_cases)
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])


def _drop_types() -> None:
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


@pytest.fixture
def ind_cls():
    _make_types(8)
    yield creator.__dict__[IND]
    _drop_types()


def test_case_halving_stages_geometric_ladder():
    assert tools.case_halving_stages(16, eta=2, min_cases=1) == (1, 2, 4, 8, 16)


def test_case_halving_stages_single_case():
    assert tools.case_halving_stages(1) == (1,)


def test_case_halving_stages_rejects_low_eta():
    with pytest.raises(ValueError, match="eta"):
        tools.case_halving_stages(4, eta=1)


def test_case_eval_charge_multiplies_counts():
    assert tools.case_eval_charge(5, 4) == 20


def test_subset_evaluate_cases_extracts_indices():
    _make_types(4)
    try:
        ind_cls = creator.__dict__[IND]
        individual = ind_cls([0])

        def evaluate(_ind):
            return (1.0, 2.0, 3.0, 4.0)

        evaluate_cases = tools.subset_evaluate_cases(evaluate)
        assert evaluate_cases(individual, [0, 2]) == (1.0, 3.0)
    finally:
        _drop_types()


def test_evaluate_case_halving_eliminates_to_one_survivor():
    _make_types(8)
    try:
        ind_cls = creator.__dict__[IND]
        train_cases = list(range(4))
        population = [ind_cls([rank]) for rank in range(4)]

        def evaluate_cases(ind, cases):
            rank = ind[0]
            return tuple(float(rank + case) for case in cases)

        result = tools.evaluate_case_halving(
            population,
            evaluate_cases,
            train_cases,
            n_cases=8,
            eta=2,
            min_cases=1,
            evaluate_full=lambda ind: tuple(float(ind[0] + case) for case in range(8)),
        )
        assert len(result.survivors) == 1
        assert result.survivors[0][0] == 0
        assert result.stages_run == len(tools.case_halving_stages(4))
    finally:
        _drop_types()


def test_evaluate_case_halving_charges_each_rung():
    _make_types(4)
    try:
        ind_cls = creator.__dict__[IND]
        train_cases = [0, 1, 2, 3]
        population = [ind_cls([0]), ind_cls([1]), ind_cls([2]), ind_cls([3])]
        stages = tools.case_halving_stages(4, eta=2, min_cases=1)
        calls: list[tuple[int, int]] = []

        def evaluate_cases(ind, cases):
            calls.append((ind[0], len(cases)))
            return tuple(float(ind[0]) for _ in cases)

        result = tools.evaluate_case_halving(
            population,
            evaluate_cases,
            train_cases,
            n_cases=4,
            eta=2,
            evaluate_full=lambda ind: tuple(float(ind[0]) for _ in range(4)),
        )
        expected = (
            tools.case_eval_charge(4, stages[0])
            + tools.case_eval_charge(2, stages[1])
            + tools.case_eval_charge(1, stages[2])
        )
        assert result.nevals == expected
        assert len(calls) == 4 + 2
    finally:
        _drop_types()


def test_evaluate_case_halving_assigns_full_catalog_fitness():
    _make_types(5)
    try:
        ind_cls = creator.__dict__[IND]
        train_cases = [0, 1, 2]
        population = [ind_cls([0]), ind_cls([1])]

        def evaluate_cases(_ind, cases):
            return tuple(0.0 for _ in cases)

        result = tools.evaluate_case_halving(
            population,
            evaluate_cases,
            train_cases,
            n_cases=5,
            eta=2,
            evaluate_full=lambda _ind: (0.1, 0.2, 0.3, 0.4, 0.5),
        )
        assert len(result.survivors[0].fitness.values) == 5
    finally:
        _drop_types()


def test_evaluate_case_halving_empty_input():
    result = tools.evaluate_case_halving([], lambda _i, _c: (), [0, 1], n_cases=2)
    assert result.survivors == []
    assert result.nevals == 0


def test_evaluate_case_halving_single_individual():
    _make_types(4)
    try:
        ind_cls = creator.__dict__[IND]
        train_cases = [0, 1, 2, 3]
        population = [ind_cls([0])]

        def evaluate_cases(_ind, cases):
            return tuple(1.0 for _ in cases)

        result = tools.evaluate_case_halving(
            population,
            evaluate_cases,
            train_cases,
            n_cases=4,
            evaluate_full=lambda _ind: (1.0, 2.0, 3.0, 4.0),
        )
        assert len(result.survivors) == 1
        stages = tools.case_halving_stages(4, eta=2, min_cases=1)
        expected = (
            tools.case_eval_charge(1, stages[0])
            + tools.case_eval_charge(1, stages[1])
            + tools.case_eval_charge(1, stages[2])
        )
        assert result.nevals == expected
    finally:
        _drop_types()
