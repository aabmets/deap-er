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

FIT = "NEVAL_FIT"
IND = "NEVAL_IND"


def _evaluate(individual):
    return (sum(individual),)


@pytest.fixture
def toolbox():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    tb = Toolbox()
    tb.register("mate", tools.cx_two_point)
    tb.register("mutate", tools.mut_flip_bit, mut_prob=1.0)
    tb.register("select", tools.sel_tournament, contestants=2)
    tb.register("evaluate", _evaluate)
    yield tb
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _unevaluated(count=4, genes=4):
    tools.rng.seed(3)
    population = []
    for _ in range(count):
        population.append(creator.__dict__[IND]([tools.rng.randint(0, 1) for _ in range(genes)]))
    return population


def _always_mutate(individual):
    individual[0] = 1 - individual[0]
    if individual.fitness.is_valid():
        del individual.fitness.values
    return (individual,)


def _drivers(toolbox, population, generations, n_evals=None, mut_prob=0.0):
    return {
        "ea_simple": lambda: tools.ea_simple(
            toolbox, population, generations, 0.0, mut_prob, n_evals=n_evals
        ),
        "ea_mu_plus_lambda": lambda: tools.ea_mu_plus_lambda(
            toolbox, population, generations, 4, 4, 0.0, mut_prob, n_evals=n_evals
        ),
        "ea_mu_comma_lambda": lambda: tools.ea_mu_comma_lambda(
            toolbox, population, generations, 4, 4, 0.0, mut_prob, n_evals=n_evals
        ),
    }


@pytest.mark.parametrize("name", ["ea_simple", "ea_mu_plus_lambda", "ea_mu_comma_lambda"])
def test_n_evals_stops_after_generation_zero(toolbox, name):
    population = _unevaluated()
    _, logbook = _drivers(toolbox, population, generations=8, n_evals=4)[name]()

    assert logbook.select("gen") == [0]
    assert logbook.select("nevals") == [4]
    assert all(ind.fitness.is_valid() for ind in population)


@pytest.mark.parametrize("name", ["ea_simple", "ea_mu_plus_lambda", "ea_mu_comma_lambda"])
def test_n_evals_none_keeps_generation_stop(toolbox, name):
    population = _unevaluated()
    _, logbook = _drivers(toolbox, population, generations=2)[name]()

    assert logbook.select("gen") == [0, 1, 2]


@pytest.mark.parametrize("name", ["ea_simple", "ea_mu_plus_lambda", "ea_mu_comma_lambda"])
def test_n_evals_stops_after_a_later_generation(toolbox, name):
    toolbox.register("mutate", _always_mutate)
    population = _unevaluated()
    for individual in population:
        individual.fitness.values = _evaluate(individual)
    _, logbook = _drivers(toolbox, population, generations=8, n_evals=8, mut_prob=1.0)[name]()

    assert logbook.select("gen") == [0, 1, 2]
    assert sum(logbook.select("nevals")) == 8


def test_n_evals_counts_eval_cache_hits_as_assignments(toolbox):
    calls = []

    def evaluate(individual):
        calls.append(1)
        return _evaluate(individual)

    cache = tools.EvalCache(evaluate)
    toolbox.register("evaluate", cache.evaluate)
    genes = [0, 1, 0, 1]
    first = [creator.__dict__[IND](list(genes)) for _ in range(4)]
    _, logbook = tools.ea_simple(toolbox, first, 0, 0.0, 0.0, n_evals=4)
    assert logbook.select("nevals") == [4]
    assert calls == [1]

    second = [creator.__dict__[IND](list(genes)) for _ in range(4)]
    _, logbook = tools.ea_simple(toolbox, second, 0, 0.0, 0.0, n_evals=4)
    assert logbook.select("nevals") == [4]
    assert all(ind.fitness.is_valid() for ind in second)
    assert calls == [1]


def test_n_evals_rejects_a_negative_budget(toolbox):
    population = _unevaluated()
    with pytest.raises(ValueError, match="at least 0"):
        tools.ea_simple(toolbox, population, 2, 0.0, 0.0, n_evals=-1)


def test_ea_map_elites_n_evals_stops_after_seeding(toolbox):
    archive = tools.GridArchive(ranges=[(4.0, 5.0)], bins=2)
    initial = _unevaluated()

    _, logbook = tools.ea_map_elites(
        toolbox,
        archive,
        lambda individual: (float(len(individual)),),
        initial,
        generations=6,
        batch_size=4,
        cx_prob=0.0,
        mut_prob=0.0,
        n_evals=4,
    )

    assert logbook.select("gen") == [0]
    assert logbook.select("nevals") == [4]


def test_ea_map_elites_n_evals_stops_after_variation(toolbox):
    toolbox.register("mutate", _always_mutate)
    archive = tools.GridArchive(ranges=[(4.0, 5.0)], bins=2)
    initial = _unevaluated()
    for individual in initial:
        individual.fitness.values = _evaluate(individual)

    _, logbook = tools.ea_map_elites(
        toolbox,
        archive,
        lambda individual: (float(len(individual)),),
        initial,
        generations=6,
        batch_size=4,
        cx_prob=0.0,
        mut_prob=1.0,
        n_evals=8,
    )

    assert logbook.select("gen") == [0, 1, 2]
    assert sum(logbook.select("nevals")) == 8
