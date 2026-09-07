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
from deap_er.private.records.grid_archive_helpers import MAX_GRID_CELLS

GA_FIT = "GAE_FIT"
GA_IND = "GAE_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(GA_FIT, Fitness, weights=(1.0,))
    creator.create_type(GA_IND, list, fitness=creator.__dict__[GA_FIT])
    yield creator.__dict__[GA_IND]
    del creator.__dict__[GA_FIT]
    del creator.__dict__[GA_IND]


def _individual(ind_cls, genes, fitness):
    individual = ind_cls(genes)
    individual.fitness.values = (fitness,)
    return individual


def test_empty_ranges_and_invalid_bins_raise():
    with pytest.raises(ValueError, match="at least one"):
        tools.GridArchive(ranges=[], bins=2)
    with pytest.raises(ValueError, match="at least 1"):
        tools.GridArchive(ranges=[(0.0, 1.0)], bins=0)
    bad_bins: Any = "4"
    with pytest.raises(ValueError, match="integer or a sequence"):
        tools.GridArchive(ranges=[(0.0, 1.0)], bins=bad_bins)
    with pytest.raises(ValueError, match="match the number"):
        tools.GridArchive(ranges=[(0.0, 1.0), (0.0, 1.0)], bins=[2])
    with pytest.raises(ValueError, match="each bin"):
        tools.GridArchive(ranges=[(0.0, 1.0), (0.0, 1.0)], bins=[2, 0])


def test_oversized_grid_raises():
    with pytest.raises(ValueError, match="exceeds"):
        tools.GridArchive(ranges=[(0.0, 1.0), (0.0, 1.0)], bins=int(MAX_GRID_CELLS))


def test_descriptor_and_index_length_errors():
    archive = tools.GridArchive(ranges=[(0.0, 1.0), (0.0, 1.0)], bins=2)

    with pytest.raises(ValueError, match="descriptor length"):
        archive.descriptor_to_index((0.1,))
    with pytest.raises(ValueError, match="index length"):
        archive.index_to_descriptor_center((0,))
    with pytest.raises(ValueError, match="out of range"):
        archive.index_to_descriptor_center((0, 9))


def test_ranges_property_and_random_elites_edges(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 2.0)], bins=4)

    assert archive.ranges == ((0.0, 2.0),)
    assert archive.random_elites(0) == []
    with pytest.raises(ValueError, match="non-negative"):
        archive.random_elites(-1)

    archive.add(_individual(ind_cls, [0], 1.0), (0.1,))
    sampled = archive.random_elites(1, replace=False)
    assert len(sampled) == 1


def test_elite_at_rejects_wrong_length_and_non_finite(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)
    archive.add(_individual(ind_cls, [0], 1.0), (0.1,))

    with pytest.raises(ValueError, match="descriptor length"):
        archive.elite_at((0.1, 0.2))
    assert archive.elite_at((math.nan,)) is None


def test_stats_skips_invalid_fitness(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=4)
    elite = _individual(ind_cls, [0], 2.0)
    archive.add(elite, (0.1,))
    stored = archive.elite_at((0.1,))
    assert stored is not None
    del stored.fitness.values

    stats = archive.stats
    assert stats.num_elites == 1
    assert stats.qd_score == 0.0
