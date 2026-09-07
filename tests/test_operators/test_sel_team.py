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
import numpy
import pytest
from deap_er import Fitness, creator, tools

THREE_FIT = "TEAM_THREE_FIT"
THREE_IND = "TEAM_THREE_IND"


@pytest.fixture
def three_cases():
    creator.create_type(THREE_FIT, Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create_type(THREE_IND, list, fitness=creator.__dict__[THREE_FIT])
    yield creator.__dict__[THREE_IND]
    del creator.__dict__[THREE_FIT]
    del creator.__dict__[THREE_IND]


def _cover_pool(make, three_cases):
    return [
        make(three_cases, [0], (0.0, 1.0, 1.0)),
        make(three_cases, [1], (1.0, 0.0, 1.0)),
        make(three_cases, [2], (1.0, 1.0, 0.0)),
        make(three_cases, [3], (0.0, 0.0, 1.0)),
    ]


def test_sel_team_k1_picks_widest_cover(three_cases, make):
    specialist, *_, generalist = _cover_pool(make, three_cases)
    population = [specialist, generalist]

    chosen = tools.sel_team(population, 1)

    assert chosen == [generalist]


def test_sel_team_greedy_adds_complementary_member(three_cases, make):
    first, second, third, generalist = _cover_pool(make, three_cases)
    population = [first, second, third, generalist]

    chosen = tools.sel_team(population, 2)

    assert chosen[0] is generalist
    assert chosen[1] is third


def test_sel_team_without_replacement_and_caps_at_pool(three_cases, make):
    population = _cover_pool(make, three_cases)

    chosen = tools.sel_team(population, 8)

    assert len(chosen) == 4
    assert len(set(map(id, chosen))) == 4
    assert all(ind in population for ind in chosen)


def test_sel_team_non_positive_count_is_empty(three_cases, make):
    population = _cover_pool(make, three_cases)

    assert tools.sel_team(population, 0) == []
    assert tools.sel_team(population, -1) == []


def test_sel_team_empty_pool_raises_index_error():
    with pytest.raises(IndexError):
        tools.sel_team([], 1)


def test_sel_team_k1_does_not_crash_on_empty_or_unsolved(single_obj, three_cases, make):
    unevaluated = single_obj([0])
    unsolved = [
        make(three_cases, [0], (1.0, 1.0, 1.0)),
        make(three_cases, [1], (2.0, 2.0, 2.0)),
    ]

    assert tools.sel_team([unevaluated], 1) == [unevaluated]
    assert tools.sel_team(unsolved, 1)[0] in unsolved


def test_sel_team_does_not_overwrite_fitness(three_cases, make):
    population = _cover_pool(make, three_cases)
    before = [ind.fitness.values for ind in population]

    tools.sel_team(population, 2)

    assert [ind.fitness.values for ind in population] == before


def test_sel_team_matrix_matches_default_path(three_cases, make):
    population = _cover_pool(make, three_cases)
    matrix = tools.fitness_case_matrix(population)

    for seed in range(16):
        tools.rng.seed(seed)
        default = tools.sel_team(population, 3)
        tools.rng.seed(seed)
        packed = tools.sel_team(population, 3, matrix=matrix)
        assert default == packed


def test_sel_team_matrix_wrong_shape_raises(three_cases, make):
    population = _cover_pool(make, three_cases)

    with pytest.raises(ValueError, match="shape"):
        tools.sel_team(population, 1, matrix=numpy.zeros((1, 3)))


def test_sel_team_matrix_mismatch_raises(three_cases, make):
    population = _cover_pool(make, three_cases)
    matrix = numpy.ones((4, 3))

    with pytest.raises(ValueError, match="does not match"):
        tools.sel_team(population, 1, matrix=matrix)


def test_sel_team_trust_matrix_skips_value_check(three_cases, make, monkeypatch):
    first, *_ = _cover_pool(make, three_cases)
    population = [first]
    matrix = numpy.array([[9.0, 8.0, 7.0]])

    def _spy_pack(*_args, **_kwargs):
        raise AssertionError("fitness_case_matrix should not run during trust validation")

    monkeypatch.setattr(
        "deap_er.private.operators.sel_lexicase_matrix.fitness_case_matrix",
        _spy_pack,
    )

    chosen = tools.sel_team(population, 1, matrix=matrix, trust_matrix=True)

    assert chosen == [first]


def test_sel_team_trust_matrix_changes_winner(three_cases, make):
    wider = make(three_cases, [0], (0.0, 0.0, 1.0))
    narrower = make(three_cases, [1], (1.0, 1.0, 0.0))
    population = [wider, narrower]
    matrix = numpy.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

    assert tools.sel_team(population, 1) == [wider]
    chosen = tools.sel_team(population, 1, matrix=matrix, trust_matrix=True)
    assert chosen == [narrower]
    with pytest.raises(ValueError, match="does not match"):
        tools.sel_team(population, 1, matrix=matrix)


def test_sel_team_duplicate_cases_covered_once(three_cases, make):
    only_zero = make(three_cases, [0], (0.0, 1.0, 1.0))
    only_one = make(three_cases, [1], (1.0, 0.0, 1.0))
    population = [only_zero, only_one]
    winners = set()
    for seed in range(40):
        tools.rng.seed(seed)
        winners.add(tools.sel_team(population, 1, cases=[0, 0, 1])[0][0])

    assert winners == {0, 1}


def test_sel_team_does_not_mutate_caller_cases(three_cases, make):
    population = _cover_pool(make, three_cases)
    cases = [2, 0, 2]

    tools.sel_team(population, 2, cases=cases)

    assert cases == [2, 0, 2]


def test_sel_team_cases_restrict_coverage(three_cases, make):
    first, second, third, generalist = _cover_pool(make, three_cases)
    population = [first, second, third, generalist]

    chosen = tools.sel_team(population, 1, cases=[2])

    assert chosen == [third]


def test_sel_team_empty_cases_still_returns_members(three_cases, make):
    population = _cover_pool(make, three_cases)

    chosen = tools.sel_team(population, 2, cases=[])

    assert len(chosen) == 2
    assert len(set(map(id, chosen))) == 2


def test_sel_team_invalid_case_raises(three_cases, make):
    population = _cover_pool(make, three_cases)

    with pytest.raises(IndexError):
        tools.sel_team(population, 1, cases=[9])


def test_sel_team_near_zero_is_solved(three_cases, make):
    almost = make(three_cases, [0], (1e-12, 1.0, 1.0))
    residual = make(three_cases, [1], (1e-11, 1.0, 1.0))
    population = [almost, residual]

    assert tools.sel_team(population, 1) == [almost]


def test_sel_team_seeded_ties_are_reproducible(three_cases, make):
    twins = [
        make(three_cases, [0], (0.0, 1.0, 1.0)),
        make(three_cases, [1], (0.0, 1.0, 1.0)),
    ]

    tools.rng.seed(3)
    first = tools.sel_team(twins, 1)
    tools.rng.seed(3)
    second = tools.sel_team(twins, 1)

    assert first == second
