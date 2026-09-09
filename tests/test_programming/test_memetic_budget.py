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
import operator

import pytest
from deap_er import Fitness, creator, gp, tools

FIT = "MEMETIC_BUDGET_FIT"
IND = "MEMETIC_BUDGET_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, gp.PrimitiveTree, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _float_pset(name: str) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_ephemeral_constant(name, lambda: 0.25)
    return pset


def _two_leaf_tree(ind_cls, pset: gp.PrimitiveSet):
    eph = pset.terminals[object][-1]
    return ind_cls([pset.mapping["add"], eph(), pset.mapping["ARG0"]])


def _target_evaluate(pset: gp.PrimitiveSet, target: float):
    def evaluate(individual):
        func = gp.compile_tree(individual, pset)
        err = func(0.0) - target
        return (err * err,)

    return evaluate


def test_estimate_tune_ephemerals_evals_matches_policy_guard():
    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    assert gp.estimate_tune_ephemerals_evals(strategy, 3) == 12
    assert (
        tools.estimate_policy_action_evals(
            "tune_ephemerals",
            strategy=strategy,
            n_gen=3,
        )
        == 12
    )


def test_cap_tune_n_gen_without_budget_caps_max_only():
    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    assert gp.cap_tune_n_gen(strategy, 9) == gp.MEMETIC_MAX_N_GEN


def test_cap_tune_n_gen_returns_zero_when_budget_spent():
    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    assert gp.cap_tune_n_gen(strategy, 2, n_evals=8, nevals_used=8) == 0


def test_cap_tune_n_gen_floors_to_affordable_generations():
    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    assert gp.cap_tune_n_gen(strategy, 5, n_evals=10, nevals_used=2) == 2


def test_tune_ephemerals_budget_defaults_to_memetic_default_n_gen(ind_cls):
    pset = _float_pset("MEMETIC_BUDGET_DEFAULT")
    tree = _two_leaf_tree(ind_cls, pset)
    gp.assign_ephemerals(tree, [0.0])
    strategy = tools.Strategy([0.0], 0.8, offsprings=2, survivors=1)
    calls = {"n_gen": 0}

    def evaluate(individual):
        calls["n_gen"] += 1
        return _target_evaluate(pset, 3.0)(individual)

    tools.rng.seed(3)
    tuned, spent = gp.tune_ephemerals_budget(tree, strategy, evaluate)
    assert tuned is tree
    assert spent == gp.estimate_tune_ephemerals_evals(strategy, gp.MEMETIC_DEFAULT_N_GEN)
    assert calls["n_gen"] == 2 * gp.MEMETIC_DEFAULT_N_GEN


def test_tune_ephemerals_budget_no_op_when_budget_exhausted(ind_cls):
    pset = _float_pset("MEMETIC_BUDGET_SPENT")
    tree = _two_leaf_tree(ind_cls, pset)
    gp.assign_ephemerals(tree, [1.5])
    before = list(tree)
    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    tuned, spent = gp.tune_ephemerals_budget(
        tree,
        strategy,
        _target_evaluate(pset, 3.0),
        n_evals=4,
        nevals_used=4,
    )
    assert tuned is tree
    assert spent == 0
    assert list(tree) == before


def test_tune_ephemerals_budget_requires_held_out_judge(ind_cls):
    pset = _float_pset("MEMETIC_BUDGET_HELD")
    tree = _two_leaf_tree(ind_cls, pset)
    strategy = tools.Strategy([0.0], 0.8, offsprings=2, survivors=1)
    train = tools.CaseExam.from_cases([1, 3], 4)
    held = tools.CaseExam.from_cases([2], 4)
    pool = tools.CaseExamPool([train], held_out=held)
    with pytest.raises(ValueError, match="held_out_evaluate"):
        gp.tune_ephemerals_budget(
            tree,
            strategy,
            _target_evaluate(pset, 3.0),
            exams=pool,
        )


def test_tune_ephemerals_budget_reports_zero_evals_without_numeric_leaves(ind_cls):
    pset = gp.PrimitiveSet("main", 1)
    tree = ind_cls([pset.mapping["ARG0"]])
    tree.fitness.values = (1.0,)
    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    calls = {"count": 0}

    def evaluate(_individual):
        calls["count"] += 1
        return (0.0,)

    tuned, spent = gp.tune_ephemerals_budget(tree, strategy, evaluate, n_gen=3)
    assert tuned is tree
    assert spent == 0
    assert calls["count"] == 0
    assert tree.fitness.is_valid()


def test_cap_tune_n_gen_rejects_negative_nevals_used():
    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    with pytest.raises(ValueError, match="nevals_used"):
        gp.cap_tune_n_gen(strategy, 2, nevals_used=-1)


def test_cap_tune_n_gen_rejects_invalid_max_n_gen():
    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    with pytest.raises(ValueError, match="max_n_gen"):
        gp.cap_tune_n_gen(strategy, 2, max_n_gen=0)


def test_resolve_tune_held_out_prefers_explicit_marker():
    train = tools.CaseExam.from_cases([0, 1], 4)
    pool_held = tools.CaseExam.from_cases([2], 4)
    explicit = tools.CaseExam.from_cases([3], 4)
    pool = tools.CaseExamPool([train], held_out=pool_held)
    from deap_er.private.programming.memetic_budget import resolve_tune_held_out

    assert resolve_tune_held_out(pool, held_out=explicit) is explicit
    assert resolve_tune_held_out(pool) is pool_held
    assert resolve_tune_held_out(train) is None


def test_tune_ephemerals_budget_requires_judge_for_explicit_held_out(ind_cls):
    pset = _float_pset("MEMETIC_BUDGET_EXPLICIT_HELD")
    tree = _two_leaf_tree(ind_cls, pset)
    strategy = tools.Strategy([0.0], 0.8, offsprings=2, survivors=1)
    held = tools.CaseExam.from_cases([2], 4)
    with pytest.raises(ValueError, match="held_out_evaluate"):
        gp.tune_ephemerals_budget(
            tree,
            strategy,
            _target_evaluate(pset, 3.0),
            held_out=held,
        )


def test_tune_ephemerals_budget_uses_held_out_evaluate_batch(ind_cls):
    pset = _float_pset("MEMETIC_BUDGET_HELD_BATCH")
    tree = _two_leaf_tree(ind_cls, pset)
    gp.assign_ephemerals(tree, [0.0])
    strategy = tools.Strategy([0.0], 0.8, offsprings=2, survivors=1)
    train = tools.CaseExam.from_cases([1, 3], 4)
    held = tools.CaseExam.from_cases([2], 4)
    pool = tools.CaseExamPool([train], held_out=held)
    used = {"train": 0, "held": 0}

    def train_batch(_individuals):
        used["train"] += 1
        return [(99.0,)] * len(_individuals)

    def held_out_batch(_individuals):
        used["held"] += 1
        return [(1.0,)] * len(_individuals)

    tools.rng.seed(9)
    tuned, spent = gp.tune_ephemerals_budget(
        tree,
        strategy,
        evaluate_batch=train_batch,
        exams=pool,
        held_out_evaluate_batch=held_out_batch,
        n_gen=1,
    )
    assert tuned is tree
    assert spent == 2
    assert used["train"] == 0
    assert used["held"] == 1


def test_tune_ephemerals_budget_uses_held_out_judge(ind_cls):
    pset = _float_pset("MEMETIC_BUDGET_JUDGE")
    tree = _two_leaf_tree(ind_cls, pset)
    gp.assign_ephemerals(tree, [0.0])
    strategy = tools.Strategy([0.0], 0.8, offsprings=2, survivors=1)
    train = tools.CaseExam.from_cases([1, 3], 4)
    held = tools.CaseExam.from_cases([2], 4)
    pool = tools.CaseExamPool([train], held_out=held)
    used = {"train": 0, "held": 0}

    def train_evaluate(_individual):
        used["train"] += 1
        return (99.0,)

    def held_out_evaluate(_individual):
        used["held"] += 1
        return (1.0,)

    tools.rng.seed(5)
    tuned, spent = gp.tune_ephemerals_budget(
        tree,
        strategy,
        train_evaluate,
        exams=pool,
        held_out_evaluate=held_out_evaluate,
        n_gen=1,
    )
    assert tuned is tree
    assert spent == 2
    assert used["train"] == 0
    assert used["held"] == 2
