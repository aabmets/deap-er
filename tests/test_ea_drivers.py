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
from deap_er import base, creator, tools

EA_FIT = "EA_FIT"
EA_IND = "EA_IND"


def _evaluate(individual):
    return (sum(individual),)


@pytest.fixture
def toolbox():
    creator.create(EA_FIT, base.Fitness, weights=(-1.0,))
    creator.create(EA_IND, list, fitness=creator.__dict__[EA_FIT])

    tb = base.Toolbox()
    tb.register("mate", tools.cx_two_point)
    tb.register("mutate", tools.mut_flip_bit, mut_prob=0.2)
    tb.register("select", tools.sel_tournament, contestants=3)
    tb.register("evaluate", _evaluate)

    yield tb

    del creator.__dict__[EA_FIT]
    del creator.__dict__[EA_IND]


def _population(count=10):
    tools.seed(3)
    population = []
    for _ in range(count):
        ind = creator.__dict__[EA_IND]([tools.rng.randint(0, 1) for _ in range(6)])
        ind.fitness.values = _evaluate(ind)
        population.append(ind)
    return population


def test_generate_update_numbers_generations_from_one(toolbox):
    strategy_pop = _population()

    def generate():
        return [toolbox.clone(ind) for ind in strategy_pop]

    def update(_population):
        return None

    toolbox.register("generate", generate)
    toolbox.register("update", update)

    _, logbook = tools.ea_generate_update(toolbox, generations=4)

    assert logbook.select("gen") == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "driver",
    [
        lambda tb, pop: tools.ea_simple(tb, pop, 4, 0.5, 0.2),
        lambda tb, pop: tools.ea_mu_plus_lambda(tb, pop, 4, 10, 10, 0.5, 0.2),
        lambda tb, pop: tools.ea_mu_comma_lambda(tb, pop, 4, 10, 10, 0.5, 0.2),
    ],
    ids=["ea_simple", "ea_mu_plus_lambda", "ea_mu_comma_lambda"],
)
def test_all_drivers_agree_on_generation_numbering(toolbox, driver):
    _, logbook = driver(toolbox, _population())

    assert logbook.select("gen") == [0, 1, 2, 3, 4]


def test_evaluate_batch_takes_over_from_map_when_registered(toolbox):
    batches = []

    def evaluate_batch(individuals):
        batches.append(len(individuals))
        return [_evaluate(ind) for ind in individuals]

    def forbidden_map(*_args, **_kwargs):
        raise AssertionError("map must not be used while evaluate_batch is registered")

    toolbox.register("evaluate_batch", evaluate_batch)
    toolbox.register("map", forbidden_map)

    _, logbook = tools.ea_simple(toolbox, _population(), 3, 0.5, 0.2)

    assert sum(batches) == sum(logbook.select("nevals"))
    assert len(batches) == len(logbook.select("gen"))


def test_map_stays_in_charge_without_an_evaluate_batch_operator(toolbox):
    mapped = []

    def counting_map(func, items):
        values = list(map(func, items))
        mapped.append(len(values))
        return values

    toolbox.register("map", counting_map)

    _, logbook = tools.ea_simple(toolbox, _population(), 3, 0.5, 0.2)

    assert sum(mapped) == sum(logbook.select("nevals"))


def test_mu_comma_lambda_rejects_more_survivors_than_offsprings(toolbox):
    # (mu, lambda) requires lambda >= mu, so this is a caller mistake
    # rather than something to silently correct.
    population = _population()
    with pytest.raises(ValueError, match="less than or equal"):
        tools.ea_mu_comma_lambda(toolbox, population, 2, 4, 8, 0.5, 0.2)


def test_mu_comma_lambda_accepts_equal_counts(toolbox):
    _, logbook = tools.ea_mu_comma_lambda(toolbox, _population(), 2, 6, 6, 0.5, 0.2)

    assert logbook.select("gen") == [0, 1, 2]
