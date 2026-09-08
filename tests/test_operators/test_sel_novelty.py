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

FIT = "NOV_FIT"
IND = "NOV_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _individual(ind_cls, genes, fitness):
    ind = ind_cls(list(genes))
    ind.fitness.values = (fitness,)
    return ind


def _descriptor(individual):
    return (float(individual[0]), float(individual[1]))


def _seed_archive(archive, ind_cls, points):
    for genes, fitness in points:
        ind = _individual(ind_cls, genes, fitness)
        archive.add(ind, _descriptor(ind))


def test_sel_novelty_picks_farthest_from_archive(ind_cls):
    archive = tools.UnstructuredArchive(2, min_distance=0.01, max_elites=8)
    _seed_archive(
        archive,
        ind_cls,
        [
            ([0.0, 0.0], 1.0),
            ([1.0, 0.0], 1.0),
        ],
    )
    near = _individual(ind_cls, [0.1, 0.0], 2.0)
    far = _individual(ind_cls, [5.0, 5.0], 2.0)
    chosen = tools.sel_novelty([near, far], 1, archive, _descriptor, k=1)
    assert chosen == [far]


def test_sel_novelty_empty_archive_falls_back_to_random(ind_cls, monkeypatch):
    population = [_individual(ind_cls, [0.0, 0.0], 1.0), _individual(ind_cls, [1.0, 1.0], 1.0)]
    archive = tools.UnstructuredArchive(2, min_distance=0.5)
    monkeypatch.setattr(tools, "sel_random", lambda pool, count: [pool[1]])
    chosen = tools.sel_novelty(population, 1, archive, _descriptor)
    assert chosen == [population[1]]


def test_sel_novelty_non_positive_count_returns_empty(ind_cls):
    archive = tools.UnstructuredArchive(2, min_distance=0.5)
    population = [_individual(ind_cls, [0.0, 0.0], 1.0)]
    assert tools.sel_novelty(population, 0, archive, _descriptor) == []
    assert tools.sel_novelty(population, -2, archive, _descriptor) == []


def test_sel_novelty_empty_pool_returns_empty(ind_cls):
    archive = tools.UnstructuredArchive(2, min_distance=0.5)
    assert tools.sel_novelty([], 3, archive, _descriptor) == []


def test_sel_novelty_rejects_non_positive_k(ind_cls):
    archive = tools.UnstructuredArchive(2, min_distance=0.5)
    population = [_individual(ind_cls, [0.0, 0.0], 1.0)]
    with pytest.raises(ValueError, match="k must be at least 1"):
        tools.sel_novelty(population, 1, archive, _descriptor, k=0)


def test_sel_novelty_does_not_overwrite_fitness(ind_cls):
    archive = tools.UnstructuredArchive(2, min_distance=0.01, max_elites=8)
    _seed_archive(archive, ind_cls, [([0.0, 0.0], 1.0)])
    candidate = _individual(ind_cls, [3.0, 3.0], 9.0)
    before = candidate.fitness.values
    tools.sel_novelty([candidate], 1, archive, _descriptor)
    assert candidate.fitness.values == before


def test_sel_novelty_unstructured_archive_uses_stored_descriptors(ind_cls):
    archive = tools.UnstructuredArchive(1, min_distance=0.01, max_elites=4)
    archive.add(_individual(ind_cls, [999.0], 1.0), (0.0,))
    near = _individual(ind_cls, [0.1], 2.0)
    far = _individual(ind_cls, [5.0], 2.0)
    chosen = tools.sel_novelty(
        [near, far],
        1,
        archive,
        lambda ind: (float(ind[0]),),
        k=1,
    )
    assert chosen == [far]


def test_sel_novelty_grid_archive_builds_matrix_from_elites(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 10.0), (0.0, 10.0)], bins=4)
    archive.add(_individual(ind_cls, [0.0, 0.0], 1.0), (0.0, 0.0))
    near = _individual(ind_cls, [0.1, 0.1], 2.0)
    far = _individual(ind_cls, [9.0, 9.0], 2.0)
    chosen = tools.sel_novelty([near, far], 1, archive, _descriptor, k=1)
    assert chosen == [far]


def test_sel_novelty_respects_valid_mask(ind_cls):
    archive = tools.UnstructuredArchive(2, min_distance=0.01, max_elites=8)
    _seed_archive(archive, ind_cls, [([1.0, 2.0], 1.0)])
    masked = _individual(ind_cls, [99.0, 2.0], 2.0)
    aligned = _individual(ind_cls, [1.0, 5.0], 2.0)
    valid = numpy.array([False, True])
    chosen = tools.sel_novelty([masked, aligned], 1, archive, _descriptor, k=1, valid=valid)
    assert chosen == [aligned]


def test_sel_novelty_ties_prefer_lowest_index(ind_cls):
    archive = tools.UnstructuredArchive(2, min_distance=0.01, max_elites=4)
    _seed_archive(archive, ind_cls, [([0.0, 0.0], 1.0)])
    first = _individual(ind_cls, [2.0, 0.0], 1.0)
    second = _individual(ind_cls, [-2.0, 0.0], 1.0)
    chosen = tools.sel_novelty([first, second], 2, archive, _descriptor, k=1)
    assert chosen == [first, second]
