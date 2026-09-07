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

US_FIT = "US_FIT"
US_IND = "US_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(US_FIT, Fitness, weights=(1.0,))
    creator.create_type(US_IND, list, fitness=creator.__dict__[US_FIT])
    yield creator.__dict__[US_IND]
    del creator.__dict__[US_FIT]
    del creator.__dict__[US_IND]


def _individual(ind_cls, genes: list[int], fitness: float):
    individual = ind_cls(genes)
    individual.fitness.values = (fitness,)
    return individual


def test_first_add_always_inserts(ind_cls):
    archive = tools.UnstructuredArchive(2, min_distance=0.5)

    assert archive.add(_individual(ind_cls, [0], 1.0), (0.0, 0.0)) is True
    assert len(archive) == 1
    assert archive.stats.num_elites == 1
    assert archive.stats.num_cells == 1
    assert archive.stats.coverage == 1.0


def test_far_candidate_opens_a_niche(ind_cls):
    archive = tools.UnstructuredArchive(2, min_distance=0.5)
    archive.add(_individual(ind_cls, [0], 1.0), (0.0, 0.0))

    assert archive.add(_individual(ind_cls, [1], 1.0), (1.0, 0.0)) is True
    assert len(archive) == 2


def test_near_better_candidate_replaces_neighbor(ind_cls):
    archive = tools.UnstructuredArchive(2, min_distance=0.5)
    archive.add(_individual(ind_cls, [0], 1.0), (0.0, 0.0))

    assert archive.add(_individual(ind_cls, [1], 2.0), (0.1, 0.0)) is True
    assert len(archive) == 1
    elite = archive.elite_at((0.0, 0.0))
    assert elite is not None
    assert elite[0] == 1
    assert archive.descriptors[0, 0] == pytest.approx(0.1)


def test_near_worse_candidate_is_rejected(ind_cls):
    archive = tools.UnstructuredArchive(2, min_distance=0.5)
    archive.add(_individual(ind_cls, [0], 2.0), (0.0, 0.0))

    assert archive.add(_individual(ind_cls, [1], 1.0), (0.1, 0.0)) is False
    elite = archive.elite_at((0.0, 0.0))
    assert elite is not None
    assert elite[0] == 0


def test_near_equal_fitness_keeps_incumbent(ind_cls):
    archive = tools.UnstructuredArchive(2, min_distance=0.5)
    first = _individual(ind_cls, [0], 2.0)
    archive.add(first, (0.0, 0.0))

    assert archive.add(_individual(ind_cls, [9], 2.0), (0.1, 0.0)) is False
    stored = archive.elite_at((0.0, 0.0))
    assert stored is not None
    assert list(stored) == list(first)


def test_max_elites_does_not_grow(ind_cls):
    archive = tools.UnstructuredArchive(2, min_distance=0.5, max_elites=1)
    archive.add(_individual(ind_cls, [0], 1.0), (0.0, 0.0))

    assert archive.add(_individual(ind_cls, [1], 0.5), (10.0, 0.0)) is False
    assert len(archive) == 1
    assert archive.add(_individual(ind_cls, [2], 3.0), (10.0, 0.0)) is True
    assert len(archive) == 1
    elite = archive.elite_at((10.0, 0.0))
    assert elite is not None
    assert elite[0] == 2
    assert archive.stats.num_cells == 1
    assert archive.stats.coverage == 1.0


def test_stats_qd_score(ind_cls):
    archive = tools.UnstructuredArchive(1, min_distance=1.0)
    archive.add(_individual(ind_cls, [0], 1.0), (0.0,))
    archive.add(_individual(ind_cls, [1], 4.0), (2.0,))

    assert archive.stats.qd_score == 5.0


def test_empty_coverage_is_zero():
    archive = tools.UnstructuredArchive(2, min_distance=0.5)
    assert archive.stats.coverage == 0.0
    assert archive.stats.num_cells == 0
    assert archive.elite_at((0.0, 0.0)) is None


def test_constructor_rejects_bad_parameters():
    with pytest.raises(ValueError, match="dimensions"):
        tools.UnstructuredArchive(0, min_distance=0.5)
    with pytest.raises(ValueError, match="min_distance"):
        tools.UnstructuredArchive(2, min_distance=0.0)
    with pytest.raises(ValueError, match="max_elites"):
        tools.UnstructuredArchive(2, min_distance=0.5, max_elites=0)


def test_add_rejects_non_finite_fitness(ind_cls):
    archive = tools.UnstructuredArchive(1, min_distance=0.5)
    archive.add(_individual(ind_cls, [0], 2.0), (0.0,))
    assert archive.add(_individual(ind_cls, [9], math.nan), (10.0,)) is False
    empty = tools.UnstructuredArchive(1, min_distance=0.5)
    assert empty.add(_individual(ind_cls, [1], math.inf), (0.0,)) is False


def test_add_rejects_non_finite_descriptor(ind_cls):
    archive = tools.UnstructuredArchive(1, min_distance=0.5)
    individual = _individual(ind_cls, [0], 1.0)

    assert archive.add(individual, (math.nan,)) is False
    assert archive.elite_at((math.inf,)) is None
    assert len(archive) == 0


def test_add_rejects_missing_or_invalid_fitness(ind_cls):
    archive = tools.UnstructuredArchive(1, min_distance=0.5)

    class Bare(list[Any]):
        pass

    assert archive.add(Bare([0]), (0.1,)) is False
    invalid = ind_cls([1])
    assert archive.add(invalid, (0.1,)) is False


def test_add_rejects_multi_objective_fitness():
    creator.create_type("US_MO_FIT", Fitness, weights=(1.0, 1.0))
    creator.create_type("US_MO_IND", list, fitness=creator.__dict__["US_MO_FIT"])
    try:
        archive = tools.UnstructuredArchive(1, min_distance=0.5)
        individual = creator.__dict__["US_MO_IND"]([0])
        individual.fitness.values = (1.0, 2.0)
        with pytest.raises(ValueError, match="single-objective"):
            archive.add(individual, (0.1,))
    finally:
        del creator.__dict__["US_MO_FIT"]
        del creator.__dict__["US_MO_IND"]


def test_wrong_descriptor_length_raises(ind_cls):
    archive = tools.UnstructuredArchive(2, min_distance=0.5)
    individual = _individual(ind_cls, [0], 1.0)
    with pytest.raises(ValueError, match="dimensions"):
        archive.add(individual, (0.1,))


def test_random_elites_on_empty_archive_raises():
    archive = tools.UnstructuredArchive(1, min_distance=0.5)
    with pytest.raises(IndexError):
        archive.random_elites(1)


def test_random_elites_without_replacement_requires_enough_elites(ind_cls):
    archive = tools.UnstructuredArchive(1, min_distance=0.5)
    archive.add(_individual(ind_cls, [0], 1.0), (0.0,))

    with pytest.raises(ValueError):
        archive.random_elites(2, replace=False)


def test_random_elites_returns_copies_not_live_references(ind_cls):
    archive = tools.UnstructuredArchive(1, min_distance=0.5)
    archive.add(_individual(ind_cls, [0], 1.0), (0.0,))

    sampled = archive.random_elites(1)[0]
    sampled[0] = 99
    elite = archive.elite_at((0.0,))
    assert elite is not None
    assert elite[0] == 0


def test_add_copies_caller_descriptor_array(ind_cls):
    archive = tools.UnstructuredArchive(2, min_distance=0.5)
    descriptor = numpy.array([0.0, 0.0], dtype=numpy.float64)
    archive.add(_individual(ind_cls, [0], 1.0), descriptor)

    descriptor[0] = 99.0

    assert archive.descriptors[0, 0] == 0.0
    elite = archive.elite_at((0.0, 0.0))
    assert elite is not None
    assert elite[0] == 0


def test_clear_and_iteration(ind_cls):
    archive = tools.UnstructuredArchive(1, min_distance=1.0)
    archive.add(_individual(ind_cls, [0], 1.0), (0.0,))
    archive.add(_individual(ind_cls, [1], 2.0), (2.0,))

    assert len(list(archive)) == 2
    archive.clear()
    assert len(archive) == 0
    assert archive.descriptors.shape == (0, 1)
