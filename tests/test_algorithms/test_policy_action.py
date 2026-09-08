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
from unittest import mock

import numpy
import pytest
from deap_er import Fitness, Toolbox, creator, gp, tools

POLICY_FIT = "POLICY_FIT"
POLICY_IND = "POLICY_IND"
POLICY_GP_FIT = "POLICY_GP_FIT"
POLICY_GP_IND = "POLICY_GP_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(POLICY_FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(POLICY_IND, list, fitness=creator.__dict__[POLICY_FIT])
    yield creator.__dict__[POLICY_IND]
    del creator.__dict__[POLICY_FIT]
    del creator.__dict__[POLICY_IND]


@pytest.fixture
def gp_ind_cls():
    creator.create_type(POLICY_GP_FIT, Fitness, weights=(-1.0,))
    creator.create_type(POLICY_GP_IND, gp.PrimitiveTree, fitness=creator.__dict__[POLICY_GP_FIT])
    yield creator.__dict__[POLICY_GP_IND]
    del creator.__dict__[POLICY_GP_FIT]
    del creator.__dict__[POLICY_GP_IND]


def test_supported_actions_match_skip_and_dispatch_tokens():
    assert tools.POLICY_ACTION_SKIP_TUNE in tools.SKIP_POLICY_ACTIONS
    assert tools.POLICY_ACTION_SKIP_PROMOTE in tools.SKIP_POLICY_ACTIONS
    assert tools.SKIP_POLICY_ACTIONS <= tools.SUPPORTED_POLICY_ACTIONS


def test_unknown_action_is_rejected():
    result = tools.apply_policy_action("load_column")
    assert result.applied is False
    assert result.rejected is True
    assert result.value is None


def test_skip_actions_noop_without_calling_underlying():
    with (
        mock.patch("deap_er.private.algorithms.policy_action.tune_ephemerals") as tune,
        mock.patch("deap_er.private.algorithms.policy_action.promote_subtree") as promote,
    ):
        tune_result = tools.apply_policy_action(tools.POLICY_ACTION_SKIP_TUNE)
        promote_result = tools.apply_policy_action(tools.POLICY_ACTION_SKIP_PROMOTE)
    assert tune_result == tools.PolicyActionResult(applied=False, rejected=False)
    assert promote_result == tools.PolicyActionResult(applied=False, rejected=False)
    tune.assert_not_called()
    promote.assert_not_called()


def test_missing_required_kwargs_are_rejected():
    result = tools.apply_policy_action("next_lexicase_cases", elites=[])
    assert result.rejected is True
    assert result.applied is False


def test_underlying_callable_errors_propagate(ind_cls):
    """Schema rejection is only for unknown tokens and missing kwargs."""
    exam = tools.CaseExam.from_cases([0], 1)
    with pytest.raises(ValueError, match="individuals must be non-empty"):
        tools.apply_policy_action(
            "next_lexicase_cases",
            exams=[exam],
            elites=[],
            mut_prob=0.0,
        )


@mock.patch("deap_er.private.algorithms.policy_action.next_lexicase_cases", return_value=[0, 2])
def test_next_lexicase_cases_dispatch(mock_next, ind_cls):
    elites = [ind_cls([0]), ind_cls([1])]
    for elite in elites:
        elite.fitness.values = (0.0, 1.0)
    exam = tools.CaseExam.from_cases([0, 1], 2)
    result = tools.apply_policy_action(
        "next_lexicase_cases",
        exams=[exam],
        elites=elites,
        mut_prob=0.0,
    )
    mock_next.assert_called_once()
    assert result.applied is True
    assert result.rejected is False
    assert result.value == [0, 2]


@mock.patch("deap_er.private.algorithms.policy_action.tune_ephemerals", return_value="tuned")
def test_tune_ephemerals_dispatch(mock_tune):
    result = tools.apply_policy_action(
        "tune_ephemerals",
        individual="tree",
        strategy="strategy",
        evaluate=lambda _: (1.0,),
        n_gen=2,
    )
    mock_tune.assert_called_once_with(
        "tree",
        "strategy",
        evaluate=mock.ANY,
        n_gen=2,
        evaluate_batch=None,
        clone=None,
    )
    assert result.applied is True
    assert result.value == "tuned"


def test_tune_ephemerals_rejects_without_evaluate_callable():
    result = tools.apply_policy_action(
        "tune_ephemerals",
        individual="tree",
        strategy="strategy",
    )
    assert result.rejected is True


@mock.patch("deap_er.private.algorithms.policy_action.promote_subtree", return_value="promo0")
def test_promote_subtree_dispatch(mock_promote):
    pset = gp.PrimitiveSetTyped("main", [float, float], float)
    pset.add_primitive(operator.add, [float, float], float)
    tree = gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset)
    result = tools.apply_policy_action(
        "promote_subtree",
        prim_set=pset,
        expr=tree,
        index=0,
        weight=2.0,
    )
    mock_promote.assert_called_once_with(
        pset,
        tree,
        index=0,
        max_library=32,
        prefix="promo",
        weight=2.0,
    )
    assert result.applied is True
    assert result.value == "promo0"


@mock.patch("deap_er.private.algorithms.policy_action.evaluate_invalid", return_value=3)
def test_evaluate_invalid_dispatch(mock_evaluate, ind_cls):
    toolbox = Toolbox()
    population = [ind_cls([0]), ind_cls([1])]
    result = tools.apply_policy_action(
        "evaluate_invalid",
        toolbox=toolbox,
        individuals=population,
    )
    mock_evaluate.assert_called_once_with(toolbox, population)
    assert result.applied is True
    assert result.value == 3


@mock.patch(
    "deap_er.private.algorithms.policy_action.interpret_tapes",
    return_value=numpy.array([[1.0, 2.0]]),
)
def test_interpret_tapes_dispatch(mock_interpret):
    matrix = numpy.array([[0.0, 1.0], [2.0, 3.0]])
    result = tools.apply_policy_action(
        "interpret_tapes",
        tapes=["tape"],
        matrix=matrix,
        backend="opcode",
    )
    mock_interpret.assert_called_once_with(
        ["tape"],
        matrix,
        backend="opcode",
        dispatch=None,
        parallel=False,
    )
    assert result.applied is True
    assert numpy.array_equal(result.value, numpy.array([[1.0, 2.0]]))


@mock.patch("deap_er.private.algorithms.policy_action.step_islands")
def test_step_islands_dispatch(mock_step, ind_cls):
    def migrate(populations):
        populations[0][:] = populations[1][:1]

    toolbox = Toolbox()
    first = [ind_cls([0])]
    second = [ind_cls([1])]
    demes = [(toolbox, first), (toolbox, second)]
    result = tools.apply_policy_action(
        "step_islands",
        demes=demes,
        migrate=migrate,
        eval_keys=("a", "b"),
    )
    mock_step.assert_called_once_with(demes, migrate, eval_keys=("a", "b"))
    assert result.applied is True
    assert result.rejected is False
    assert result.value is None


def test_next_lexicase_cases_integration(ind_cls):
    elites = [ind_cls([0]), ind_cls([1])]
    for elite in elites:
        elite.fitness.values = (0.0, 1.0)
    exam = tools.CaseExam.from_cases([0, 1], 2)
    result = tools.apply_policy_action(
        "next_lexicase_cases",
        exams=[exam],
        elites=elites,
        mut_prob=0.0,
    )
    assert result.applied is True
    assert result.value == [0, 1]


def test_evaluate_invalid_integration(ind_cls):
    toolbox = Toolbox()

    def evaluate(individual):
        return (float(individual[0]), 0.0)

    toolbox.register("evaluate", evaluate)
    population = [ind_cls([3.0])]
    result = tools.apply_policy_action(
        "evaluate_invalid",
        toolbox=toolbox,
        individuals=population,
    )
    assert result.applied is True
    assert result.value == 1
    assert population[0].fitness.values == (3.0, 0.0)


def test_promote_subtree_integration():
    pset = gp.PrimitiveSetTyped("main", [float, float], float)
    pset.add_primitive(operator.add, [float, float], float)
    tree = gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset)
    result = tools.apply_policy_action(
        "promote_subtree",
        prim_set=pset,
        expr=tree,
    )
    assert result.applied is True
    assert result.value == "promo0"
    assert "promo0" in gp.promoted_names(pset)


def _memetic_pset(name: str) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_ephemeral_constant(name, lambda: 0.25)
    return pset


def _memetic_tree(gp_ind_cls, pset: gp.PrimitiveSet):
    eph = pset.terminals[object][-1]
    return gp_ind_cls([pset.mapping["add"], eph(), pset.mapping["ARG0"]])


def test_tune_ephemerals_integration(gp_ind_cls):
    pset = _memetic_pset("POLICY_MEMETIC")
    tree = _memetic_tree(gp_ind_cls, pset)
    gp.assign_ephemerals(tree, [0.0])
    tree.fitness.values = (99.0,)

    def evaluate(individual):
        func = gp.compile_tree(individual, pset)
        err = func(0.0) - 3.0
        return (err * err,)

    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    tools.rng.seed(7)
    result = tools.apply_policy_action(
        "tune_ephemerals",
        individual=tree,
        strategy=strategy,
        evaluate=evaluate,
        n_gen=1,
    )
    assert result.applied is True
    assert result.value is tree
    assert not tree.fitness.is_valid()


def test_interpret_tapes_integration():
    pset = gp.make_column_pset(["value"])
    gp.add_numpy_primitives(pset)
    pset.add_terminal(2.5, gp.Array, "two_half")
    tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["two_half"]]), pset)
    matrix = numpy.array([[1.0], [2.0], [3.0]])
    result = tools.apply_policy_action(
        "interpret_tapes",
        tapes=[tape],
        matrix=matrix,
    )
    assert result.applied is True
    numpy.testing.assert_allclose(result.value[0], 2.5)


def test_step_islands_integration():
    fit_name = "POLICY_ISLAND_FIT"
    ind_name = "POLICY_ISLAND_IND"
    creator.create_type(fit_name, Fitness, weights=(1.0,))
    creator.create_type(ind_name, list, fitness=creator.__dict__[fit_name])
    ind_cls = creator.__dict__[ind_name]

    def vary(population):
        return list(population) + [ind_cls([10.0]), ind_cls([11.0])]

    toolbox = Toolbox()
    toolbox.register("evaluate", lambda ind: (float(ind[0]),))
    toolbox.register("vary", vary)
    toolbox.register("select", tools.sel_best)
    population = [ind_cls([0.0]), ind_cls([1.0])]
    try:
        result = tools.apply_policy_action(
            "step_islands",
            demes=[(toolbox, population)],
        )
    finally:
        del creator.__dict__[fit_name]
        del creator.__dict__[ind_name]

    assert result.applied is True
    assert result.rejected is False
    assert sorted(ind[0] for ind in population) == [10.0, 11.0]
