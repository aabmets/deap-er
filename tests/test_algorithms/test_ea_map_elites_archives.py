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
from deap_er import Fitness, Toolbox, creator, tools

ME_FIT = "MEA_FIT"
ME_IND = "MEA_IND"


def _evaluate(individual):
    return (sum(individual),)


def _behavior(individual):
    return (float(len(individual)),)


@pytest.fixture
def toolbox():
    creator.create_type(ME_FIT, Fitness, weights=(1.0,))
    creator.create_type(ME_IND, list, fitness=creator.__dict__[ME_FIT])

    tb = Toolbox()
    tb.register("mate", tools.cx_two_point)
    tb.register("mutate", tools.mut_flip_bit, mut_prob=0.2)
    tb.register("evaluate", _evaluate)

    yield tb

    del creator.__dict__[ME_FIT]
    del creator.__dict__[ME_IND]


def _population(count=20, gene_len=6):
    tools.rng.seed(7)
    population = []
    for index in range(count):
        genes = [tools.rng.randint(0, 1) for _ in range(gene_len + (index % 3))]
        individual = creator.__dict__[ME_IND](genes)
        population.append(individual)
    return population


def _cvt_archive():
    return tools.CvtArchive([[6.0], [7.0], [8.0], [9.0]])


def _unstructured_archive():
    return tools.UnstructuredArchive(1, min_distance=0.75, max_elites=8)


@pytest.mark.parametrize("factory", [_cvt_archive, _unstructured_archive])
def test_ea_map_elites_accepts_cvt_and_unstructured(toolbox, factory):
    archive = factory()
    initial = _population()

    tools.ea_map_elites(
        toolbox,
        archive,
        _behavior,
        initial,
        generations=4,
        batch_size=12,
        cx_prob=0.5,
        mut_prob=0.2,
    )

    assert len(archive) >= 1
    assert archive.stats.num_elites >= 1
    assert archive.stats.coverage > 0.0


@pytest.mark.parametrize("factory", [_cvt_archive, _unstructured_archive])
def test_ea_map_elites_logbook_columns_for_new_archives(toolbox, factory):
    archive = factory()
    initial = _population(count=8)

    _, logbook = tools.ea_map_elites(
        toolbox,
        archive,
        _behavior,
        initial,
        generations=2,
        batch_size=8,
        cx_prob=0.0,
        mut_prob=0.2,
    )

    assert logbook.select("gen") == [0, 1, 2]
    assert logbook.select("coverage")
    assert logbook.select("num_elites")
    assert logbook.select("qd_score")
