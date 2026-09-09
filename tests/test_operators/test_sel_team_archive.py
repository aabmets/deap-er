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

ARCH_FIT = "TEAM_ARCH_FIT"
ARCH_IND = "TEAM_ARCH_IND"


@pytest.fixture
def archive_types():
    creator.create_type(ARCH_FIT, Fitness, weights=(1.0,))
    creator.create_type(ARCH_IND, list, fitness=creator.__dict__[ARCH_FIT])
    yield creator.__dict__[ARCH_IND]
    del creator.__dict__[ARCH_FIT]
    del creator.__dict__[ARCH_IND]


def _specialists(make, ind_cls):
    first = make(ind_cls, [0], (0.4,))
    second = make(ind_cls, [1], (0.5,))
    third = make(ind_cls, [2], (0.6,))
    generalist = make(ind_cls, [3], (0.2,))
    matrix = numpy.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=numpy.float64,
    )
    descriptors = [
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    return first, second, third, generalist, matrix, descriptors


def _grid_archive(make, ind_cls):
    first, second, third, generalist, matrix, descriptors = _specialists(make, ind_cls)
    archive = tools.GridArchive(ranges=[(0.0, 1.0)] * 3, bins=2)
    for individual, descriptor in zip(
        [first, second, third, generalist],
        descriptors,
        strict=True,
    ):
        archive.add(individual, descriptor)
    return archive, matrix, generalist


def _solved_columns(matrix: numpy.ndarray) -> int:
    solved = 0
    for col in range(matrix.shape[1]):
        if numpy.any(numpy.isclose(matrix[:, col], 0.0, atol=1e-12)):
            solved += 1
    return solved


def test_sel_team_archive_greedy_cover(make, archive_types):
    first, second, third, _, matrix, descriptors = _specialists(make, archive_types)
    archive = tools.GridArchive(ranges=[(0.0, 1.0)] * 3, bins=2)
    for individual, descriptor in zip([first, second, third], descriptors[:3], strict=True):
        archive.add(individual, descriptor)
    specialist_matrix = matrix[:3]
    best_single = max(
        _solved_columns(specialist_matrix[idx : idx + 1])
        for idx in range(specialist_matrix.shape[0])
    )

    team = tools.sel_team_archive(archive, 3, matrix=specialist_matrix, trust_matrix=True)
    team_rows = numpy.asarray(
        [specialist_matrix[list(archive).index(member)] for member in team],
        dtype=numpy.float64,
    )
    team_union = _solved_columns(team_rows)

    assert best_single == 1
    assert team_union == 3
    assert len(team) == 3


def test_sel_team_archive_empty_raises():
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=2)

    with pytest.raises(IndexError):
        tools.sel_team_archive(archive, 1)


def test_sel_team_archive_does_not_rewrite_fitness(make, archive_types):
    archive, matrix, _ = _grid_archive(make, archive_types)
    before = {id(ind): ind.fitness.values for ind in archive}

    tools.sel_team_archive(archive, 2, matrix=matrix, trust_matrix=True)

    assert {id(ind): ind.fitness.values for ind in archive} == before


def test_sel_team_archive_matrix_kwarg(make, archive_types, monkeypatch):
    archive, matrix, _ = _grid_archive(make, archive_types)

    def _spy_team(*args, **kwargs):
        assert kwargs["matrix"] is matrix
        assert kwargs["trust_matrix"] is True
        return tools.sel_team(*args, **kwargs)

    monkeypatch.setattr("deap_er.private.operators.sel_team.sel_team", _spy_team)
    tools.sel_team_archive(archive, 1, matrix=matrix, trust_matrix=True)


def test_sel_team_archive_matches_manual_sel_team(make, archive_types):
    archive, matrix, _ = _grid_archive(make, archive_types)
    elites = list(archive)

    for seed in range(16):
        tools.rng.seed(seed)
        from_archive = tools.sel_team_archive(archive, 2, matrix=matrix, trust_matrix=True)
        tools.rng.seed(seed)
        manual = tools.sel_team(elites, 2, matrix=matrix, trust_matrix=True)
        assert from_archive == manual


def test_sel_team_archive_works_on_cvt_and_unstructured(make, archive_types):
    first, second, third, generalist, matrix, descriptors = _specialists(make, archive_types)
    centroids = numpy.asarray(descriptors[:3], dtype=numpy.float64)
    cvt = tools.CvtArchive(centroids)
    unstructured = tools.UnstructuredArchive(3, min_distance=0.05)
    for individual, descriptor in zip([first, second, third], descriptors[:3], strict=True):
        cvt.add(individual, descriptor)
        unstructured.add(individual, descriptor)

    cvt_team = tools.sel_team_archive(cvt, 2, matrix=matrix[:3], trust_matrix=True)
    unstructured_team = tools.sel_team_archive(
        unstructured,
        2,
        matrix=matrix[:3],
        trust_matrix=True,
    )

    assert len(cvt_team) == 2
    assert len(unstructured_team) == 2
    assert len({id(member) for member in cvt_team}) == 2
    assert len({id(member) for member in unstructured_team}) == 2
