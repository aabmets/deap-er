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
from deap_er import Fitness, creator, gp, records, tools

FIT = "POLICY_FIT"
IND = "POLICY_IND"


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


def test_policy_observation_schema_is_stable():
    assert records.PolicyObservation.FIELD_NAMES == (
        "solve_bits",
        "unsolved_count",
        "train_score",
        "held_out_score",
        "archive_coverage",
        "qd_score",
        "nevals",
        "rows_seen",
        "promoted_library_size",
        "fitness_invalid",
        "last_action_rejected",
    )
    obs = tools.policy_observe(solve_bits=(1, 0), train_score=3.0)
    assert isinstance(obs, records.PolicyObservation)
    assert obs.as_tuple() == (
        (1, 0),
        1,
        3.0,
        None,
        0.0,
        0.0,
        0,
        0,
        0,
        False,
        False,
    )


def test_policy_observe_rejects_raw_arrays():
    with pytest.raises(TypeError, match="solve_bits must not be a raw array"):
        tools.policy_observe(solve_bits=numpy.array([1, 0]), train_score=0.0)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="errors must not be a raw array"):
        tools.policy_solve_bits_from_errors(numpy.array([0.0, 1.0]))  # ty: ignore[invalid-argument-type]


def test_policy_solve_bits_from_case_errors():
    predicted = numpy.array([0.0, 0.0, 1.0, 2.0, 0.0, 0.0], dtype=numpy.float64)
    target = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=numpy.float64)
    ranges = [(0, 2), (2, 4), (4, 6)]
    errors = tools.case_errors(predicted, target, ranges)
    bits = tools.policy_solve_bits_from_errors(errors)
    assert bits == (1, 0, 1)
    assert tools.policy_unsolved_count(bits) == 1


def test_policy_exam_scores_from_score_case_exams(ind_cls):
    elites = [
        _elite(ind_cls, (0.0, 1.0, 0.0, 1.0)),
        _elite(ind_cls, (0.0, 0.0, 1.0, 1.0)),
    ]
    train = tools.CaseExam.from_cases([1, 3], 4)
    held = tools.CaseExam.from_cases([2], 4)
    pool = tools.CaseExamPool([train], held_out=held)
    train_score, held_out_score = tools.policy_exam_scores(pool, elites)
    assert train_score == float(sum(tools.score_case_exams([train], elites)))
    assert held_out_score == float(tools.score_case_exams([held], elites)[0])


def test_policy_observe_from_summary_surfaces(ind_cls):
    elites = [_elite(ind_cls, (0.0, 1.0, 0.0, 1.0))]
    errors = (0.0, 1.0, 0.0, 1.0)
    solve_bits = tools.policy_solve_bits_from_fitness(errors)
    held = tools.CaseExam.from_cases([1], 4)
    train = tools.CaseExam.from_cases([0, 2], 4)
    pool = tools.CaseExamPool([train], held_out=held)
    train_score, held_out_score = tools.policy_exam_scores(pool, elites)
    stats = records.ArchiveStats(
        num_elites=2,
        num_cells=4,
        coverage=0.5,
        qd_score=3.0,
    )
    pset = gp.PrimitiveSetTyped("MAIN", [float, float], float)
    pset.add_primitive(operator.add, [float, float], float)
    tree = gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset)
    gp.promote_subtree(pset, tree)
    obs = tools.policy_observe(
        solve_bits=solve_bits,
        train_score=train_score,
        held_out_score=held_out_score,
        archive=stats,
        nevals=7,
        rows_seen=128,
        promoted_library_size=tools.policy_promoted_library_size(pset),
        fitness_invalid=False,
        last_action_rejected=True,
    )
    assert obs.solve_bits == (1, 0, 1, 0)
    assert obs.unsolved_count == 2
    assert obs.train_score == train_score
    assert obs.held_out_score == held_out_score
    assert obs.archive_coverage == stats.coverage
    assert obs.qd_score == stats.qd_score
    assert obs.nevals == 7
    assert obs.rows_seen == 128
    assert obs.promoted_library_size == 1
    assert obs.last_action_rejected is True


def test_policy_solve_bits_from_semantic_row_matches_case_errors():
    matrix = numpy.array([[0.0, 0.0, 1.0, 2.0, 0.0, 0.0]], dtype=numpy.float64)
    target = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=numpy.float64)
    ranges = [(0, 2), (2, 4), (4, 6)]
    semantic = tools.semantic_solve_bits(matrix, target, ranges)
    row_bits = tools.policy_solve_bits_from_semantic_row(tuple(float(v) for v in semantic[0]))
    error_bits = tools.policy_solve_bits_from_errors(tools.case_errors(matrix[0], target, ranges))
    assert row_bits == error_bits
