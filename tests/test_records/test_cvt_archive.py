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
from typing import Any

import numpy
import pytest
from deap_er import Fitness, creator, tools

CV_FIT = "CV_FIT"
CV_IND = "CV_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(CV_FIT, Fitness, weights=(1.0,))
    creator.create_type(CV_IND, list, fitness=creator.__dict__[CV_FIT])
    yield creator.__dict__[CV_IND]
    del creator.__dict__[CV_FIT]
    del creator.__dict__[CV_IND]


def _individual(ind_cls, genes: list[int], fitness: float):
    individual = ind_cls(genes)
    individual.fitness.values = (fitness,)
    return individual


def _two_centroids():
    return numpy.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=numpy.float64)


def test_add_to_empty_cell(ind_cls):
    archive = tools.CvtArchive(_two_centroids())

    assert archive.add(_individual(ind_cls, [0], 1.0), (0.1, 0.1)) is True
    assert len(archive) == 1
    assert archive.stats.num_elites == 1
    assert archive.stats.num_cells == 2
    assert archive.stats.coverage == 0.5


def test_add_assigns_nearest_centroid(ind_cls):
    archive = tools.CvtArchive(_two_centroids())
    archive.add(_individual(ind_cls, [0], 1.0), (0.1, 0.0))
    archive.add(_individual(ind_cls, [1], 2.0), (0.9, 1.0))

    assert archive.nearest_centroid((0.1, 0.0)) == 0
    assert archive.nearest_centroid((0.9, 1.0)) == 1
    assert 0 in archive
    assert 1 in archive
    assert archive.get(0) is not None
    assert archive.get(1) is not None


def test_add_replaces_strictly_better_fitness(ind_cls):
    archive = tools.CvtArchive(_two_centroids())
    archive.add(_individual(ind_cls, [0], 1.0), (0.1, 0.1))

    assert archive.add(_individual(ind_cls, [1], 2.0), (0.05, 0.0)) is True
    elite = archive.elite_at((0.1, 0.1))
    assert elite is not None
    assert elite[0] == 1


def test_add_keeps_incumbent_on_equal_fitness(ind_cls):
    archive = tools.CvtArchive(_two_centroids())
    first = _individual(ind_cls, [0], 2.0)
    archive.add(first, (0.1, 0.1))

    assert archive.add(_individual(ind_cls, [9], 2.0), (0.0, 0.0)) is False
    stored = archive.elite_at((0.1, 0.1))
    assert stored is not None
    assert list(stored) == list(first)


def test_add_rejects_strictly_worse_fitness(ind_cls):
    archive = tools.CvtArchive(_two_centroids())
    archive.add(_individual(ind_cls, [0], 2.0), (0.1, 0.1))

    assert archive.add(_individual(ind_cls, [1], 1.0), (0.0, 0.0)) is False
    elite = archive.elite_at((0.1, 0.1))
    assert elite is not None
    assert elite[0] == 0


def test_stats_qd_score(ind_cls):
    archive = tools.CvtArchive(_two_centroids())
    archive.add(_individual(ind_cls, [0], 1.0), (0.0, 0.0))
    archive.add(_individual(ind_cls, [1], 3.0), (1.0, 1.0))

    assert archive.stats.qd_score == 4.0


def test_from_samples_and_cvt_centroids(ind_cls):
    tools.rng.seed(0)
    samples = [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]]
    centroids = tools.cvt_centroids(samples, 2, n_iter=5)
    assert centroids.shape == (2, 2)

    tools.rng.seed(0)
    archive = tools.CvtArchive.from_samples(samples, 2, n_iter=5)
    assert archive.centroids.shape == (2, 2)
    archive.add(_individual(ind_cls, [0], 1.0), (0.0, 0.0))
    assert len(archive) == 1


def test_cvt_centroids_rejects_k_greater_than_samples():
    with pytest.raises(ValueError, match="cannot exceed"):
        tools.cvt_centroids([[0.0, 0.0]], 2)


def test_cvt_centroids_rejects_k_less_than_one():
    with pytest.raises(ValueError, match="at least 1"):
        tools.cvt_centroids([[0.0, 0.0]], 0)


def test_cvt_centroids_rejects_n_iter_less_than_one():
    with pytest.raises(ValueError, match="n_iter"):
        tools.cvt_centroids([[0.0, 0.0]], 1, n_iter=0)


def test_cvt_centroids_rejects_one_dimensional_samples():
    with pytest.raises(ValueError, match="2-D"):
        tools.cvt_centroids(numpy.asarray([0.0, 1.0, 2.0]), 1)


def test_cvt_centroids_rejects_non_finite_samples():
    with pytest.raises(ValueError, match="finite"):
        tools.cvt_centroids([[0.0, math.nan]], 1)


def test_duplicate_centroids_raise():
    with pytest.raises(ValueError, match="unique"):
        tools.CvtArchive([[0.0, 0.0], [0.0, 0.0]])


def test_add_rejects_non_finite_fitness(ind_cls):
    archive = tools.CvtArchive(_two_centroids())
    archive.add(_individual(ind_cls, [0], 2.0), (0.1, 0.1))
    assert archive.add(_individual(ind_cls, [9], math.nan), (0.1, 0.1)) is False
    empty = tools.CvtArchive(_two_centroids())
    assert empty.add(_individual(ind_cls, [1], math.inf), (0.1, 0.1)) is False


def test_add_rejects_non_finite_descriptor(ind_cls):
    archive = tools.CvtArchive(_two_centroids())
    individual = _individual(ind_cls, [0], 1.0)

    assert archive.add(individual, (math.nan, 0.0)) is False
    assert archive.elite_at((math.inf, 0.0)) is None
    assert len(archive) == 0


def test_add_rejects_missing_or_invalid_fitness(ind_cls):
    archive = tools.CvtArchive(_two_centroids())

    class Bare(list[Any]):
        pass

    assert archive.add(Bare([0]), (0.1, 0.1)) is False
    invalid = ind_cls([1])
    assert archive.add(invalid, (0.1, 0.1)) is False


def test_add_rejects_multi_objective_fitness():
    creator.create_type("CV_MO_FIT", Fitness, weights=(1.0, 1.0))
    creator.create_type("CV_MO_IND", list, fitness=creator.__dict__["CV_MO_FIT"])
    try:
        archive = tools.CvtArchive(_two_centroids())
        individual = creator.__dict__["CV_MO_IND"]([0])
        individual.fitness.values = (1.0, 2.0)
        with pytest.raises(ValueError, match="single-objective"):
            archive.add(individual, (0.1, 0.1))
    finally:
        del creator.__dict__["CV_MO_FIT"]
        del creator.__dict__["CV_MO_IND"]


def test_wrong_descriptor_length_raises(ind_cls):
    archive = tools.CvtArchive(_two_centroids())
    with pytest.raises(ValueError, match="dimensions"):
        archive.add(_individual(ind_cls, [0], 1.0), (0.1,))


def test_random_elites_on_empty_archive_raises():
    archive = tools.CvtArchive(_two_centroids())
    with pytest.raises(IndexError):
        archive.random_elites(1)


def test_random_elites_without_replacement_requires_enough_elites(ind_cls):
    archive = tools.CvtArchive(_two_centroids())
    archive.add(_individual(ind_cls, [0], 1.0), (0.0, 0.0))

    with pytest.raises(ValueError):
        archive.random_elites(2, replace=False)


def test_random_elites_returns_copies_not_live_references(ind_cls):
    archive = tools.CvtArchive(_two_centroids())
    archive.add(_individual(ind_cls, [0], 1.0), (0.1, 0.1))

    sampled = archive.random_elites(1)[0]
    sampled[0] = 99
    elite = archive.elite_at((0.1, 0.1))
    assert elite is not None
    assert elite[0] == 0


def test_constructor_copies_caller_centroid_array(ind_cls):
    centroids = numpy.array([[0.0, 0.0], [1.0, 1.0]], dtype=numpy.float64)
    archive = tools.CvtArchive(centroids)
    archive.add(_individual(ind_cls, [0], 1.0), (0.1, 0.0))

    centroids[0, 0] = 99.0

    assert archive.centroids[0, 0] == 0.0
    assert archive.nearest_centroid((0.1, 0.0)) == 0
    assert 0 in archive


def test_clear_and_iteration(ind_cls):
    archive = tools.CvtArchive(_two_centroids())
    archive.add(_individual(ind_cls, [0], 1.0), (0.0, 0.0))
    archive.add(_individual(ind_cls, [1], 2.0), (1.0, 1.0))

    assert len(list(archive)) == 2
    archive.clear()
    assert len(archive) == 0
