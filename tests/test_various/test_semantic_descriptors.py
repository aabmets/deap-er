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
import math

import numpy
import pytest
from deap_er import Fitness, creator, gp, tools

FIT = "SEM_DESC_FIT"
IND = "SEM_DESC_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _individual(ind_cls, fitness: float):
    individual = ind_cls([])
    individual.fitness.values = (fitness,)
    return individual


def _row_moments(row: numpy.ndarray, valid: numpy.ndarray | None = None) -> list[float]:
    mask = numpy.isfinite(row) if valid is None else valid & numpy.isfinite(row)
    if not numpy.any(mask):
        return [math.nan, math.nan, math.nan, math.nan]
    values = row[mask]
    return [float(values.mean()), float(values.std(ddof=0)), float(values.min()), float(values.max())]


def test_semantic_moments_match_row_oracle():
    matrix = numpy.array([[1.0, 2.0, 3.0], [4.0, 8.0, 8.0]])
    actual = tools.semantic_moments(matrix)
    expected = numpy.array([_row_moments(row) for row in matrix])
    numpy.testing.assert_allclose(actual, expected)


def test_semantic_moments_exclude_warmup_nans_and_valid():
    matrix = numpy.array([[math.nan, 2.0, 4.0], [0.0, 0.0, 6.0]])
    valid = numpy.array([False, True, True])
    actual = tools.semantic_moments(matrix, valid=valid)
    expected = numpy.array([_row_moments(row, valid) for row in matrix])
    numpy.testing.assert_allclose(actual, expected)


def test_semantic_moments_empty_row_is_nan():
    matrix = numpy.array([[math.nan, math.nan]])
    actual = tools.semantic_moments(matrix)
    assert actual.shape == (1, 4)
    assert numpy.all(numpy.isnan(actual))


def test_semantic_solve_bits_match_case_errors():
    matrix = numpy.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    target = numpy.zeros(4)
    ranges = [(0, 2), (2, 4)]
    actual = tools.semantic_solve_bits(matrix, target, ranges)
    bits = []
    for row in matrix:
        errors = tools.case_errors(row, target, ranges)
        bits.append([math.isclose(error, 0.0, abs_tol=1e-12) for error in errors])
    numpy.testing.assert_array_equal(actual, numpy.asarray(bits, dtype=numpy.float64))


def test_semantic_project_and_random_pca_shapes(ind_cls):
    tools.rng.seed(7)
    matrix = numpy.array([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]])
    basis = tools.semantic_random_basis(4, 2)
    projected = tools.semantic_project(matrix, basis)
    pca_basis, center = tools.semantic_pca_basis(matrix, 2)
    pca = tools.semantic_project(matrix, pca_basis, center=center)
    people = [_individual(ind_cls, 1.0), _individual(ind_cls, 2.0)]
    trusted = tools.semantic_descriptors(
        matrix, kind="moments", individuals=people, trust_matrix=True
    )
    assert basis.shape == (4, 2)
    assert projected.shape == (2, 2)
    assert pca_basis.shape == (4, 2)
    assert pca.shape == (2, 2)
    assert trusted.shape == (2, 4)


def test_semantic_descriptors_dispatch_and_trust(ind_cls):
    matrix = numpy.array([[1.0, 1.0, 1.0, 1.0]])
    target = numpy.ones(4)
    people = [_individual(ind_cls, 0.5)]
    moments = gp.semantic_descriptors(matrix, kind="moments")
    solved = gp.semantic_descriptors(
        matrix, kind="solve", target=target, ranges=[(0, 2), (2, 4)], individuals=people
    )
    basis = numpy.eye(4)[:, :1]
    projected = gp.semantic_descriptors(matrix, kind="project", basis=basis)
    numpy.testing.assert_allclose(moments, [[1.0, 0.0, 1.0, 1.0]])
    numpy.testing.assert_array_equal(solved, [[1.0, 1.0]])
    numpy.testing.assert_allclose(projected, [[1.0]])


def test_semantic_valid_mask_rejects_two_dimensional_valid():
    matrix = numpy.zeros((2, 3))
    with pytest.raises(ValueError, match="one-dimensional"):
        tools.semantic_valid_mask(matrix, numpy.ones((2, 3), dtype=bool))


def test_semantic_descriptors_rejects_unknown_or_incomplete_kind():
    matrix = numpy.zeros((1, 2))
    with pytest.raises(ValueError, match="unknown descriptor kind"):
        tools.semantic_descriptors(matrix, kind="latent")
    with pytest.raises(ValueError, match="target and ranges"):
        tools.semantic_descriptors(matrix, kind="solve")
    with pytest.raises(ValueError, match="requires basis"):
        tools.semantic_descriptors(matrix, kind="project")


def test_semantic_trust_matrix_requires_valid_fitness(ind_cls):
    matrix = numpy.zeros((1, 3))
    bare = ind_cls([])
    with pytest.raises(ValueError, match="valid fitness"):
        tools.semantic_moments(matrix, individuals=[bare], trust_matrix=False)
    trusted = tools.semantic_moments(matrix, individuals=[bare], trust_matrix=True)
    assert trusted.shape == (1, 4)
