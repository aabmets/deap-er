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

import numpy
import pytest
from deap_er import Fitness, Toolbox, creator, gp, tools

POL_FIT = "EA_POLICY_FIT"
POL_IND = "EA_POLICY_IND"
N_CASES = 4


@pytest.fixture
def toolbox():
    creator.create_type(POL_FIT, Fitness, weights=(-1.0,) * N_CASES)
    creator.create_type(POL_IND, list, fitness=creator.__dict__[POL_FIT])
    tb = Toolbox()
    tb.register("mate", tools.cx_two_point)
    tb.register("mutate", tools.mut_flip_bit, mut_prob=0.2)
    tb.register("select", tools.sel_tournament, contestants=2)
    tb.register("evaluate", lambda individual: tuple(float(gene) for gene in individual))
    yield tb
    del creator.__dict__[POL_FIT]
    del creator.__dict__[POL_IND]


def _population():
    tools.rng.seed(3)
    population = []
    for _ in range(8):
        genes = [tools.rng.randint(0, 1) for _ in range(N_CASES)]
        individual = creator.__dict__[POL_IND](genes)
        population.append(individual)
    return population


def _pool():
    train = tools.CaseExam.from_cases([0, 1], N_CASES)
    held = tools.CaseExam.from_cases([2, 3], N_CASES)
    return tools.CaseExamPool([train], held_out=held)


def _promote_pset():
    pset = gp.PrimitiveSetTyped("policy_main", [float, float], float)
    pset.add_primitive(operator.add, [float, float], float)
    return pset


def _promote_tree(pset):
    return gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset)


def test_ea_policy_records_action_and_matches_ea_simple_generations(toolbox):
    actions = []

    def decide(obs):
        actions.append(obs.last_action_rejected)
        return tools.POLICY_ACTION_SKIP_PROMOTE

    population = _population()
    _, logbook = tools.ea_policy(
        toolbox,
        population,
        decide,
        generations=2,
        cx_prob=0.5,
        mut_prob=0.2,
    )
    assert logbook.select("gen") == [0, 1, 2]
    assert logbook.header[0:3] == ["gen", "nevals", "action"]
    assert logbook.select("action") == [tools.POLICY_ACTION_SKIP_PROMOTE] * 3
    assert actions == [False, False]
    assert all(ind.fitness.is_valid() for ind in population)


def test_ea_policy_uses_exams_and_lexicase_cases(toolbox):
    seen = []

    def decide(obs):
        seen.append(obs.unsolved_count)
        return "next_lexicase_cases"

    pool = _pool()
    guard = tools.PolicyActionGuard(max_promotes_per_gen=1)
    population = _population()
    _, logbook = tools.ea_policy(
        toolbox,
        population,
        decide,
        generations=2,
        cx_prob=0.4,
        mut_prob=0.1,
        exams=pool,
        n_cases=N_CASES,
        guard=guard,
        elite_count=3,
        action_kwargs={"mut_prob": 0.0},
    )
    assert logbook.select("action")[1:] == ["next_lexicase_cases", "next_lexicase_cases"]
    chapter = logbook.chapters["generalization_gap"]
    assert len(chapter) == 3
    assert "held_out" in chapter[1]
    assert seen
    assert all(ind.fitness.is_valid() for ind in population)


def test_ea_policy_honors_n_evals(toolbox):
    def decide(_obs):
        return tools.POLICY_ACTION_SKIP_PROMOTE

    population = _population()
    _, logbook = tools.ea_policy(
        toolbox,
        population,
        decide,
        generations=5,
        cx_prob=0.5,
        mut_prob=0.2,
        n_evals=8,
    )
    assert logbook.select("gen") == [0]
    with pytest.raises(ValueError, match="n_evals"):
        tools.ea_policy(toolbox, population, decide, 1, 0.5, 0.2, n_evals=-1)


def test_ea_policy_observe_kwargs_cannot_hide_a_rejected_action(toolbox):
    seen: list[bool] = []

    def decide(obs):
        seen.append(obs.last_action_rejected)
        return "promote_subtree"

    tools.ea_policy(
        toolbox,
        _population(),
        decide,
        generations=2,
        cx_prob=0.0,
        mut_prob=0.0,
        observe_kwargs={"last_action_rejected": False},
    )
    assert seen == [False, True]


def test_ea_policy_ignores_column_matrix_for_lexicase_actions(toolbox):
    def decide(_obs):
        return "next_lexicase_cases"

    pool = _pool()
    _, logbook = tools.ea_policy(
        toolbox,
        _population(),
        decide,
        generations=1,
        cx_prob=0.0,
        mut_prob=0.0,
        exams=pool,
        n_cases=N_CASES,
        action_kwargs={"matrix": numpy.ones((8, 2)), "mut_prob": 0.0},
    )
    assert logbook.select("action")[1] == "next_lexicase_cases"


def test_ea_policy_advances_guard_generation_for_cooldown(toolbox):
    pset = _promote_pset()
    tree = _promote_tree(pset)
    seen: list[bool] = []

    def decide(obs):
        seen.append(obs.last_action_rejected)
        return "promote_subtree"

    guard = tools.PolicyActionGuard(max_promotes_per_gen=1, promote_cooldown=2)
    tools.ea_policy(
        toolbox,
        _population(),
        decide,
        generations=3,
        cx_prob=0.0,
        mut_prob=0.0,
        guard=guard,
        action_kwargs={"prim_set": pset, "expr": tree},
    )
    assert seen == [False, False, True]
    assert guard.last_promote_gen == 3
