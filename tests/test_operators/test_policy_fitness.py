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

FIT = "POLICY_FIT13"
IND = "POLICY_IND13"


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _elite(ind_cls, values):
    individual = ind_cls([0])
    individual.fitness.values = values
    return individual


def _pool_and_elites(ind_cls):
    elites = [
        _elite(ind_cls, (0.0, 1.0, 0.0, 1.0)),
        _elite(ind_cls, (0.0, 0.0, 1.0, 1.0)),
    ]
    train = tools.CaseExam.from_cases([1, 3], 4)
    held = tools.CaseExam.from_cases([2], 4)
    pool = tools.CaseExamPool([train], held_out=held)
    return pool, held, train, elites


def test_regression_held_out_only_policy_fitness(ind_cls):
    pool, held, train, elites = _pool_and_elites(ind_cls)
    fitness = tools.policy_held_out_fitness(elites, pool)
    assert fitness == float(tools.score_case_exams([held], elites)[0])
    assert fitness != float(sum(tools.score_case_exams([train], elites)))


def test_regression_train_exam_refused_as_fitness_target(ind_cls):
    pool, held, train, elites = _pool_and_elites(ind_cls)
    with pytest.raises(ValueError, match="train-exam quality"):
        tools.guard_policy_fitness_exam(
            train,
            held_out=held,
            n_cases=4,
            train_exams=pool.exams,
        )
    with pytest.raises(ValueError, match="train-exam quality"):
        tools.policy_held_out_fitness(elites, pool, fitness_exam=train)


def test_regression_mutated_held_out_refused_as_fitness_target(ind_cls):
    pool, held, _train, elites = _pool_and_elites(ind_cls)
    with pytest.raises(ValueError, match="must not reward the exam the policy just mutated"):
        tools.policy_held_out_fitness(
            elites,
            pool,
            fitness_exam=held,
            mutated_exam=held,
        )
    mutated_copy = tools.CaseExam.from_cases([2], 4)
    with pytest.raises(ValueError, match="must not reward the exam the policy just mutated"):
        tools.guard_policy_fitness_exam(
            held,
            held_out=held,
            n_cases=4,
            mutated_exam=mutated_copy,
        )


def test_policy_held_out_fitness_uses_only_held_out_exam(ind_cls):
    pool, held, train, elites = _pool_and_elites(ind_cls)
    fitness = tools.policy_held_out_fitness(elites, pool)
    assert fitness == float(tools.score_case_exams([held], elites)[0])
    assert fitness != float(sum(tools.score_case_exams([train], elites)))


def test_policy_held_out_fitness_requires_marked_held_out(ind_cls):
    elites = [_elite(ind_cls, (0.0, 1.0, 0.0, 1.0))]
    train = tools.CaseExam.from_cases([1, 3], 4)
    pool = tools.CaseExamPool([train])
    with pytest.raises(ValueError, match="held_out"):
        tools.policy_held_out_fitness(elites, pool)


def test_train_exam_quality_is_not_policy_fitness(ind_cls):
    pool, held, train, elites = _pool_and_elites(ind_cls)
    with pytest.raises(ValueError, match="train-exam quality"):
        tools.guard_policy_fitness_exam(
            train,
            held_out=held,
            n_cases=4,
            train_exams=pool.exams,
        )
    with pytest.raises(ValueError, match="train-exam quality"):
        tools.policy_held_out_fitness(
            elites,
            pool,
            fitness_exam=train,
        )


def test_policy_fitness_refuses_mutated_exam(ind_cls):
    pool, held, train, elites = _pool_and_elites(ind_cls)
    with pytest.raises(ValueError, match="must not reward the exam the policy just mutated"):
        tools.policy_held_out_fitness(
            elites,
            pool,
            fitness_exam=held,
            mutated_exam=held,
        )
    with pytest.raises(ValueError, match="train-exam quality"):
        tools.policy_held_out_fitness(
            elites,
            pool,
            fitness_exam=train,
            mutated_exam=train,
        )


def test_resolve_policy_held_out_honors_override(ind_cls):
    pool, held, train, elites = _pool_and_elites(ind_cls)
    override = tools.CaseExam.from_cases([0], 4)
    resolved = tools.resolve_policy_held_out(pool, held_out=override)
    assert resolved is override
    assert tools.policy_held_out_fitness(elites, pool, held_out=override) == float(
        tools.score_case_exams([override], elites)[0]
    )


def test_policy_exam_scores_train_is_observation_only(ind_cls):
    pool, held, train, elites = _pool_and_elites(ind_cls)
    train_score, held_out_score = tools.policy_exam_scores(pool, elites)
    fitness = tools.policy_held_out_fitness(elites, pool)
    assert fitness == held_out_score
    assert train_score != fitness
    obs = tools.policy_observe(
        solve_bits=tools.policy_solve_bits_from_fitness(elites[0].fitness.values),
        train_score=train_score,
        held_out_score=held_out_score,
    )
    assert obs.train_score == train_score
    assert obs.held_out_score == held_out_score
