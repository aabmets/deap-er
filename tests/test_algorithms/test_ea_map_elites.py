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

ME_FIT = "ME_FIT"
ME_IND = "ME_IND"


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


def test_ea_map_elites_increases_coverage(toolbox):
    archive = tools.GridArchive(ranges=[(6.0, 9.0)], bins=4)
    initial = _population()

    tools.ea_map_elites(
        toolbox,
        archive,
        _behavior,
        initial,
        generations=8,
        batch_size=20,
        cx_prob=0.5,
        mut_prob=0.2,
    )

    assert archive.stats.num_elites >= 2
    assert archive.stats.coverage > 0.0


def test_ea_map_elites_logbook_columns(toolbox):
    archive = tools.GridArchive(ranges=[(6.0, 9.0)], bins=4)
    initial = _population(count=10)

    _, logbook = tools.ea_map_elites(
        toolbox,
        archive,
        _behavior,
        initial,
        generations=3,
        batch_size=10,
        cx_prob=0.5,
        mut_prob=0.2,
    )

    assert logbook.select("gen") == [0, 1, 2, 3]
    assert logbook.select("coverage")
    assert logbook.select("num_elites")
    assert logbook.select("qd_score")


def test_ea_map_elites_evaluates_initial_before_generation_one(toolbox):
    archive = tools.GridArchive(ranges=[(6.0, 9.0)], bins=4)
    initial = _population(count=5)
    for individual in initial:
        del individual.fitness.values

    _, logbook = tools.ea_map_elites(
        toolbox,
        archive,
        _behavior,
        initial,
        generations=1,
        batch_size=5,
        cx_prob=0.0,
        mut_prob=0.0,
    )

    assert all(ind.fitness.is_valid() for ind in initial)
    assert logbook.select("nevals")[0] == 5
    assert len(archive) >= 1


def test_ea_map_elites_uses_evaluate_batch_when_registered(toolbox):
    archive = tools.GridArchive(ranges=[(6.0, 9.0)], bins=4)
    initial = _population(count=6)
    batches = []

    def evaluate_batch(individuals):
        batches.append(len(individuals))
        return [_evaluate(ind) for ind in individuals]

    def forbidden_map(*_args, **_kwargs):
        raise AssertionError("map must not be used while evaluate_batch is registered")

    toolbox.register("evaluate_batch", evaluate_batch)
    toolbox.register("map", forbidden_map)

    tools.ea_map_elites(
        toolbox,
        archive,
        _behavior,
        initial,
        generations=2,
        batch_size=6,
        cx_prob=0.0,
        mut_prob=0.0,
    )

    assert batches


def _max_stats():
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", max)
    return stats


def test_ea_map_elites_with_empty_initial_leaves_archive_empty(toolbox):
    archive = tools.GridArchive(ranges=[(0.0, 10.0)], bins=4)

    _, logbook = tools.ea_map_elites(
        toolbox,
        archive,
        _behavior,
        [],
        generations=0,
        batch_size=5,
        cx_prob=0.0,
        mut_prob=0.0,
    )

    assert len(archive) == 0
    assert logbook.select("gen") == [0]


def test_ea_map_elites_empty_initial_generation_zero_with_stats(toolbox):
    # Generation 0 still records when initial is empty. Tutorial-style
    # max stats must not reduce an empty seed list.
    archive = tools.GridArchive(ranges=[(0.0, 10.0)], bins=4)

    _, logbook = tools.ea_map_elites(
        toolbox,
        archive,
        _behavior,
        [],
        generations=0,
        batch_size=5,
        cx_prob=0.0,
        mut_prob=0.0,
        stats=_max_stats(),
    )

    assert logbook.select("gen") == [0]
    assert len(archive) == 0


def test_ea_map_elites_prefilled_archive_empty_initial_with_stats(toolbox):
    # Resume from a seeded archive: variation is allowed with initial=[].
    # Gen 0 skips stats.compile on that empty seed; later gens compile offspring.
    archive = tools.GridArchive(ranges=[(6.0, 9.0)], bins=4)
    seed = creator.__dict__[ME_IND]([0, 1, 1, 0, 1, 1])
    seed.fitness.values = _evaluate(seed)
    assert archive.add(seed, _behavior(seed))

    _, logbook = tools.ea_map_elites(
        toolbox,
        archive,
        _behavior,
        [],
        generations=2,
        batch_size=4,
        cx_prob=0.0,
        mut_prob=0.2,
        stats=_max_stats(),
    )

    assert logbook.select("gen") == [0, 1, 2]
    assert len(archive) >= 1
    assert all(value is not None for value in logbook.select("max")[1:])


def test_ea_map_elites_empty_initial_with_generations_raises(toolbox):
    archive = tools.GridArchive(ranges=[(0.0, 10.0)], bins=4)

    with pytest.raises(ValueError, match="non-empty"):
        tools.ea_map_elites(
            toolbox,
            archive,
            _behavior,
            [],
            generations=1,
            batch_size=5,
            cx_prob=0.0,
            mut_prob=0.0,
        )


def test_ea_map_elites_batch_size_one_with_crossover_does_not_crash(toolbox):
    archive = tools.GridArchive(ranges=[(6.0, 9.0)], bins=4)
    initial = _population(count=5)

    tools.ea_map_elites(
        toolbox,
        archive,
        _behavior,
        initial,
        generations=5,
        batch_size=1,
        cx_prob=0.5,
        mut_prob=0.2,
    )

    assert len(archive) >= 1
