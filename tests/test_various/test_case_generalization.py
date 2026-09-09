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

from unittest import mock

import pytest
from deap_er import Fitness, creator, tools

FIT = "GEN_FIT"
IND = "GEN_IND"


def _make_types(n_cases: int):
    creator.create_type(FIT, Fitness, weights=(-1.0,) * n_cases)
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])


def _drop_types() -> None:
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


@pytest.fixture
def ind_cls():
    _make_types(20)
    yield creator.__dict__[IND]
    _drop_types()


def test_held_out_tail_matches_last_fraction():
    assert tools.held_out_tail(20, 0.2) == [16, 17, 18, 19]


def test_train_head_complements_held_out_tail():
    n_cases = 20
    held = tools.held_out_tail(n_cases, 0.2)
    train = tools.train_head(n_cases, 0.2)
    assert sorted(train + held) == list(range(n_cases))
    assert set(train).isdisjoint(held)


def test_held_out_tail_small_catalog():
    assert tools.held_out_tail(2, 0.5) == [1]
    assert tools.train_head(2, 0.5) == [0]


def test_held_out_tail_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="fraction"):
        tools.held_out_tail(10, 0.0)
    with pytest.raises(ValueError, match="fraction"):
        tools.held_out_tail(10, 1.0)


def test_held_out_tail_rejects_invalid_n_cases():
    with pytest.raises(ValueError, match="n_cases"):
        tools.held_out_tail(0, 0.2)


def test_case_generalization_pool_default_split():
    pool = tools.case_generalization_pool(20, fraction=0.2)
    assert pool.held_out is not None
    assert pool.held_out.as_cases(20) == tools.held_out_tail(20, 0.2)
    assert pool.exams[0].as_cases(20) == tools.train_head(20, 0.2)


def test_case_generalization_pool_custom_held_cases():
    pool = tools.case_generalization_pool(8, held_cases=[2, 5])
    assert pool.held_out is not None
    assert pool.held_out.as_cases(8) == [2, 5]
    assert pool.exams[0].as_cases(8) == [0, 1, 3, 4, 6, 7]


def test_case_generalization_recipe_fields():
    recipe = tools.case_generalization_recipe(20, fraction=0.2)
    assert recipe.n_cases == 20
    assert recipe.held_cases == tuple(tools.held_out_tail(20, 0.2))
    assert recipe.train_cases == tuple(tools.train_head(20, 0.2))
    assert recipe.pool.held_out is not None


def test_make_lexicase_train_select_uses_train_cases_only():
    _make_types(6)
    try:
        ind_cls = creator.__dict__[IND]
        tools.rng.seed(0)
        pool = tools.case_generalization_pool(6, held_cases=[5])
        select = tools.make_lexicase_train_select(pool, 6)
        population = [ind_cls([i]) for i in range(4)]
        for idx, ind in enumerate(population):
            ind.fitness.values = tuple(float(idx + bit) for bit in range(6))
        with mock.patch(
            "deap_er.private.various.case_generalization.sel_lexicase",
            wraps=tools.sel_lexicase,
        ) as lexicase:
            chosen = select(population, 2)
            assert lexicase.call_args.kwargs["cases"] == [0, 1, 2, 3, 4]
        assert len(chosen) == 2
        assert 5 not in lexicase.call_args.kwargs["cases"]
    finally:
        _drop_types()


def test_make_lexicase_train_select_downsample_caps_cases():
    _make_types(10)
    try:
        ind_cls = creator.__dict__[IND]
        tools.rng.seed(1)
        pool = tools.case_generalization_pool(10, held_cases=[9])
        select = tools.make_lexicase_train_select(pool, 10, downsample=3)
        population = [ind_cls([0]) for _ in range(6)]
        for idx, ind in enumerate(population):
            ind.fitness.values = tuple(float(idx + bit) for bit in range(10))
        with mock.patch(
            "deap_er.private.various.case_generalization.sel_lexicase",
            wraps=tools.sel_lexicase,
        ) as lexicase:
            select(population, 2)
            cases = lexicase.call_args.kwargs["cases"]
        assert len(cases) <= 3
        assert 9 not in cases
    finally:
        _drop_types()


def test_generalization_recipe_make_select():
    _make_types(8)
    try:
        ind_cls = creator.__dict__[IND]
        tools.rng.seed(2)
        recipe = tools.case_generalization_recipe(8, held_cases=[7])
        select = recipe.make_select()
        population = [ind_cls([float(i)]) for i in range(5)]
        for ind in population:
            ind.fitness.values = tuple(float(i) for i in range(8))
        chosen = select(population, 1)
        assert len(chosen) == 1
    finally:
        _drop_types()
