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

HOF_FIT = "HOF_FIT"
HOF_IND = "HOF_IND"


@pytest.fixture
def ind_cls():
    creator.create(HOF_FIT, base.Fitness, weights=(1.0,))
    creator.create(HOF_IND, list, fitness=creator.__dict__[HOF_FIT])
    yield creator.__dict__[HOF_IND]
    del creator.__dict__[HOF_FIT]
    del creator.__dict__[HOF_IND]


def _population(ind_cls, count=5):
    population = []
    for i in range(count):
        ind = ind_cls([i])
        ind.fitness.values = (float(i),)
        population.append(ind)
    return population


def test_remove_from_empty_raises_index_error():
    with pytest.raises(IndexError):
        tools.HallOfFame(maxsize=3).remove(0)


def test_remove_keeps_items_and_keys_aligned(ind_cls):
    hof = tools.HallOfFame(maxsize=5)
    hof.update(_population(ind_cls))

    hof.remove(0)

    assert [ind[0] for ind in hof] == [3, 2, 1, 0]
    assert len(hof.keys) == len(hof.items)


def test_update_keeps_only_the_best(ind_cls):
    hof = tools.HallOfFame(maxsize=2)

    hof.update(_population(ind_cls))

    assert [ind[0] for ind in hof] == [4, 3]


def test_hall_of_fame_clear_reversed_and_str(ind_cls):
    hof = tools.HallOfFame(maxsize=3)
    hof.update(_population(ind_cls))

    assert list(reversed(hof))[-1] is hof[0]
    assert str(hof)
    hof.clear()
    assert len(hof) == 0


def test_pareto_front_keeps_non_dominated_and_drops_twins():
    creator.create("PF_FIT", base.Fitness, weights=(-1.0, -1.0))
    creator.create("PF_IND", list, fitness=creator.__dict__["PF_FIT"])
    try:
        front = tools.ParetoFront()

        def _ind(genes, values):
            individual = creator.__dict__["PF_IND"](genes)
            individual.fitness.values = values
            return individual

        first = _ind([0], (1.0, 4.0))
        second = _ind([1], (4.0, 1.0))
        dominated = _ind([2], (5.0, 5.0))
        twin = _ind([0], (1.0, 4.0))
        better = _ind([3], (0.5, 0.5))

        front.update([first, second, dominated, twin])
        assert len(front) == 2
        front.update([better])
        assert len(front) == 1
        assert front[0].fitness.values == (0.5, 0.5)
    finally:
        del creator.__dict__["PF_FIT"]
        del creator.__dict__["PF_IND"]
