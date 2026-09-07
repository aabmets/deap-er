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

import pytest
from deap_er import Fitness, creator, tools

GA_FIT = "GA_FIT"
GA_IND = "GA_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(GA_FIT, Fitness, weights=(1.0,))
    creator.create_type(GA_IND, list, fitness=creator.__dict__[GA_FIT])
    yield creator.__dict__[GA_IND]
    del creator.__dict__[GA_FIT]
    del creator.__dict__[GA_IND]


def _individual(ind_cls, genes: list[int], fitness: float):
    individual = ind_cls(genes)
    individual.fitness.values = (fitness,)
    return individual


def test_add_to_empty_cell(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)

    assert archive.add(_individual(ind_cls, [0], 1.0), (0.1,)) is True
    assert len(archive) == 1
    assert archive.stats.num_elites == 1


def test_add_replaces_strictly_better_fitness(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)
    archive.add(_individual(ind_cls, [0], 1.0), (0.1,))

    assert archive.add(_individual(ind_cls, [1], 2.0), (0.1,)) is True
    elite = archive.elite_at((0.1,))
    assert elite is not None
    assert elite[0] == 1
    assert elite.fitness.values == (2.0,)


def test_add_rejects_strictly_worse_fitness(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)
    archive.add(_individual(ind_cls, [0], 2.0), (0.1,))

    assert archive.add(_individual(ind_cls, [1], 1.0), (0.1,)) is False
    elite = archive.elite_at((0.1,))
    assert elite is not None
    assert elite[0] == 0


def test_add_keeps_incumbent_on_equal_fitness(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)
    first = _individual(ind_cls, [0], 2.0)
    archive.add(first, (0.1,))

    assert archive.add(_individual(ind_cls, [9], 2.0), (0.1,)) is False
    stored = archive.elite_at((0.1,))
    assert stored is not None
    assert list(stored) == list(first)


def test_descriptor_to_index_corners():
    archive = tools.GridArchive(ranges=[(0.0, 10.0)], bins=10)

    assert archive.descriptor_to_index((0.0,)) == (0,)
    assert archive.descriptor_to_index((10.0,)) == (9,)
    assert archive.descriptor_to_index((5.0,)) == (5,)


def test_descriptor_to_index_clips_out_of_range():
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)

    assert archive.descriptor_to_index((-1.0,)) == (0,)
    assert archive.descriptor_to_index((2.0,)) == (3,)


def test_two_dimensional_cells_are_distinct(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0), (0.0, 1.0)], bins=2)
    archive.add(_individual(ind_cls, [0], 1.0), (0.1, 0.1))
    archive.add(_individual(ind_cls, [1], 2.0), (0.9, 0.9))

    assert len(archive) == 2
    assert archive.get((0, 0)) is not None
    assert archive.get((1, 1)) is not None


def test_random_elites_with_replacement(ind_cls):
    tools.rng.seed(0)
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=5)
    for value in (0.1, 0.3, 0.7):
        archive.add(_individual(ind_cls, [int(value * 10)], float(value)), (value,))

    elites = archive.random_elites(6, replace=True)
    assert len(elites) == 6
    assert all(hasattr(ind, "fitness") for ind in elites)


def test_stats_coverage_and_qd_score(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=10)
    archive.add(_individual(ind_cls, [0], 1.0), (0.1,))
    archive.add(_individual(ind_cls, [1], 2.0), (0.5,))
    archive.add(_individual(ind_cls, [2], 3.0), (0.9,))

    stats = archive.stats
    assert stats.num_elites == 3
    assert stats.coverage == 0.3
    assert stats.qd_score == 6.0


def test_clear_and_iteration(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=3)
    archive.add(_individual(ind_cls, [0], 1.0), (0.1,))
    archive.add(_individual(ind_cls, [1], 2.0), (0.9,))

    assert len(list(archive)) == 2
    archive.clear()
    assert len(archive) == 0


def test_elite_at_and_get_round_trip(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)
    individual = _individual(ind_cls, [4], 4.0)
    archive.add(individual, (0.5,))

    index = archive.descriptor_to_index((0.5,))
    assert archive.get(index) is not None
    assert archive.elite_at((0.5,)) is not None
    assert (index,) == (archive.descriptor_to_index((0.5,)),)


def test_add_rejects_non_finite_fitness(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)
    archive.add(_individual(ind_cls, [0], 2.0), (0.1,))
    assert archive.add(_individual(ind_cls, [9], math.nan), (0.1,)) is False
    elite = archive.elite_at((0.1,))
    assert elite is not None
    assert elite[0] == 0
    empty = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)
    assert empty.add(_individual(ind_cls, [1], math.inf), (0.1,)) is False
    assert len(empty) == 0


def test_add_rejects_non_finite_descriptor(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)
    individual = _individual(ind_cls, [0], 1.0)

    assert archive.add(individual, (math.nan,)) is False
    assert archive.add(individual, (math.inf,)) is False
    assert len(archive) == 0


def test_add_rejects_missing_or_invalid_fitness(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)

    class Bare(list[Any]):
        pass

    assert archive.add(Bare([0]), (0.1,)) is False
    invalid = ind_cls([1])
    assert archive.add(invalid, (0.1,)) is False


def test_random_elites_on_empty_archive_raises():
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)
    with pytest.raises(IndexError):
        archive.random_elites(1)


def test_random_elites_without_replacement_requires_enough_elites(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=5)
    archive.add(_individual(ind_cls, [0], 1.0), (0.1,))
    archive.add(_individual(ind_cls, [1], 2.0), (0.9,))

    with pytest.raises(ValueError):
        archive.random_elites(3, replace=False)


def test_wrong_descriptor_length_raises(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0), (0.0, 1.0)], bins=2)
    individual = _individual(ind_cls, [0], 1.0)
    with pytest.raises(ValueError):
        archive.add(individual, (0.1,))


def test_uniform_bins_int_constructor(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0), (0.0, 1.0)], bins=3)
    archive.add(_individual(ind_cls, [0], 1.0), (0.1, 0.1))

    assert archive.bins == (3, 3)


def test_equal_range_raises():
    with pytest.raises(ValueError):
        tools.GridArchive(ranges=[(1.0, 1.0)], bins=4)


def test_stored_elite_survives_mutation_of_sample(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)
    archive.add(_individual(ind_cls, [0], 1.0), (0.1,))

    tools.rng.seed(0)
    sampled = archive.random_elites(1)[0]
    mutant = tools.clone_individual(sampled)
    mutant[0] = 99

    elite = archive.elite_at((0.1,))
    assert elite is not None
    assert elite[0] == 0


def test_index_to_descriptor_center(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 10.0)], bins=10)
    center = archive.index_to_descriptor_center((5,))

    assert center == (5.5,)


def test_contains_operator(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)
    index = archive.descriptor_to_index((0.1,))
    assert index not in archive
    archive.add(_individual(ind_cls, [0], 1.0), (0.1,))
    assert index in archive


def test_random_elites_returns_copies_not_live_references(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)
    archive.add(_individual(ind_cls, [0], 1.0), (0.1,))

    sampled = archive.random_elites(1)[0]
    sampled[0] = 99

    elite = archive.elite_at((0.1,))
    assert elite is not None
    assert elite[0] == 0


def test_add_rejects_multi_objective_fitness():
    creator.create_type("MO_FIT", Fitness, weights=(1.0, 1.0))
    creator.create_type("MO_IND", list, fitness=creator.__dict__["MO_FIT"])
    try:
        archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)
        individual = creator.__dict__["MO_IND"]([0])
        individual.fitness.values = (1.0, 2.0)
        with pytest.raises(ValueError, match="single-objective"):
            archive.add(individual, (0.1,))
    finally:
        del creator.__dict__["MO_FIT"]
        del creator.__dict__["MO_IND"]


def test_numpy_scalar_bins_are_accepted():
    numpy = pytest.importorskip("numpy")

    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=numpy.int64(5))
    assert archive.bins == (5,)
