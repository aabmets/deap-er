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
from deap_er import Fitness, Toolbox, creator, tools

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
