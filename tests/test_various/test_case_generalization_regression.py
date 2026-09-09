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

FIT = "GEN_REG_FIT"
IND = "GEN_REG_IND"


def _make_types(n_cases: int):
    creator.create_type(FIT, Fitness, weights=(-1.0,) * n_cases)
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])


def _drop_types() -> None:
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def test_lexicase_train_select_ignores_held_out_fitness():
    """Held-out catalog indices must not drive lexicase selection."""
    _make_types(6)
    try:
        ind_cls = creator.__dict__[IND]
        tools.rng.seed(0)
        pool = tools.case_generalization_pool(6, held_cases=[5])
        select = tools.make_lexicase_train_select(pool, 6)
        train_winner = ind_cls([0])
        held_winner = ind_cls([1])
        train_winner.fitness.values = (0.0, 10.0, 10.0, 10.0, 10.0, 10.0)
        held_winner.fitness.values = (10.0, 10.0, 10.0, 10.0, 10.0, 0.0)
        chosen = select([train_winner, held_winner], 1)
        assert chosen[0] is train_winner
    finally:
        _drop_types()


def test_lexicase_train_select_downsample_still_ignores_held_out():
    """Downsampled lexicase must not let held-out fitness decide winners."""
    _make_types(10)
    try:
        ind_cls = creator.__dict__[IND]
        tools.rng.seed(3)
        pool = tools.case_generalization_pool(10, held_cases=[9])
        select = tools.make_lexicase_train_select(
            pool,
            10,
            downsample=3,
            downsample_mode="held_out",
        )
        train_winner = ind_cls([0])
        held_winner = ind_cls([1])
        train_winner.fitness.values = (0.0,) + (10.0,) * 9
        held_winner.fitness.values = (10.0,) * 9 + (0.0,)
        chosen = select([train_winner, held_winner], 1)
        assert chosen[0] is train_winner
    finally:
        _drop_types()


def test_evaluate_case_halving_uses_train_case_prefix_order():
    """Halving must score caller-ordered train prefixes, not sorted indices."""
    _make_types(10)
    try:
        ind_cls = creator.__dict__[IND]
        train_cases = [3, 7, 1, 9]
        seen: list[tuple[int, ...]] = []

        def evaluate_cases(_ind, cases):
            seen.append(tuple(cases))
            return tuple(0.0 for _ in cases)

        population = [ind_cls([rank]) for rank in range(4)]
        tools.evaluate_case_halving(
            population,
            evaluate_cases,
            train_cases,
            n_cases=10,
            eta=2,
            min_cases=1,
            evaluate_full=lambda ind: tuple(float(ind[0] + i) for i in range(10)),
        )
        assert seen[0] == (3,)
        assert (3, 7) in seen
        assert seen.index((3, 7)) > seen.index((3,))
    finally:
        _drop_types()


def test_evaluate_case_halving_final_charge_uses_catalog_size():
    """Final rung charges n_cases when fitness spans the full catalog."""
    _make_types(8)
    try:
        ind_cls = creator.__dict__[IND]
        train_cases = [0, 1, 2, 3]
        population = [ind_cls([0]), ind_cls([1]), ind_cls([2]), ind_cls([3])]
        stages = tools.case_halving_stages(len(train_cases), eta=2, min_cases=1)

        def evaluate_cases(ind, cases):
            return tuple(float(ind[0]) for _ in cases)

        result = tools.evaluate_case_halving(
            population,
            evaluate_cases,
            train_cases,
            n_cases=8,
            eta=2,
            evaluate_full=lambda ind: tuple(float(ind[0] + i) for i in range(8)),
        )
        expected = (
            tools.case_eval_charge(4, stages[0])
            + tools.case_eval_charge(2, stages[1])
            + tools.case_eval_charge(1, 8)
        )
        assert result.nevals == expected
    finally:
        _drop_types()


def test_evaluate_case_halving_rejects_short_final_fitness():
    _make_types(6)
    try:
        ind_cls = creator.__dict__[IND]

        def evaluate_cases(_ind, cases):
            return tuple(0.0 for _ in cases)

        with pytest.raises(ValueError, match="final fitness must have length n_cases"):
            tools.evaluate_case_halving(
                [ind_cls([0])],
                evaluate_cases,
                [0, 1],
                n_cases=6,
                evaluate_full=lambda _ind: (1.0, 2.0, 3.0),
            )
    finally:
        _drop_types()


def test_evaluate_case_halving_assigns_full_catalog_not_train_only():
    _make_types(8)
    try:
        ind_cls = creator.__dict__[IND]
        train_cases = [0, 1, 2, 3]
        population = [ind_cls([0]), ind_cls([1])]

        def evaluate_cases(_ind, cases):
            return tuple(0.0 for _ in cases)

        result = tools.evaluate_case_halving(
            population,
            evaluate_cases,
            train_cases,
            n_cases=8,
            eta=2,
            evaluate_full=lambda _ind: tuple(float(i) for i in range(8)),
        )
        assert len(result.survivors[0].fitness.values) == 8
        assert len(result.survivors[0].fitness.values) != len(train_cases)
    finally:
        _drop_types()
