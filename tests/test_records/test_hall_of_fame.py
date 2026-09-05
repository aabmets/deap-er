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
from typing import Any

import pytest
from deap_er import Fitness, creator, tools

HOF_FIT = "HOF_FIT"
HOF_IND = "HOF_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(HOF_FIT, Fitness, weights=(1.0,))
    creator.create_type(HOF_IND, list, fitness=creator.__dict__[HOF_FIT])
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
    hof = tools.HallOfFame(maxsize=3)
    with pytest.raises(IndexError):
        hof.remove(0)


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


def test_update_replaces_similar_member_when_fitness_is_better(ind_cls):
    hof = tools.HallOfFame(maxsize=2)
    better_other = ind_cls([1])
    better_other.fitness.values = (10.0,)
    worse = ind_cls([0])
    worse.fitness.values = (5.0,)
    hof.update([better_other, worse])
    improved = ind_cls([0])
    improved.fitness.values = (8.0,)
    hof.update([improved])
    assert [(list(ind), ind.fitness.values[0]) for ind in hof] == [([1], 10.0), ([0], 8.0)]


def test_update_skips_no_fitness_bootstrap_and_keeps_later_members(ind_cls):
    class Bare(list[Any]):
        pass

    good = ind_cls([9])
    good.fitness.values = (7.0,)
    hof = tools.HallOfFame(maxsize=5)
    hof.update([Bare([0]), good])
    assert len(hof) == 1
    assert list(hof[0]) == [9]
    assert hof[0].fitness.values == (7.0,)


def test_update_with_zero_maxsize_is_a_noop(ind_cls):
    hof = tools.HallOfFame(maxsize=0)
    ind = ind_cls([1])
    ind.fitness.values = (1.0,)
    hof.update([ind])
    assert len(hof) == 0


def test_remove_out_of_range_keeps_keys_aligned(ind_cls):
    hof = tools.HallOfFame(maxsize=5)
    hof.update(_population(ind_cls, count=3))
    with pytest.raises(IndexError):
        hof.remove(99)
    assert len(hof.items) == len(hof.keys) == 3


def test_pareto_front_keeps_non_dominated_and_drops_twins():
    creator.create_type("PF_FIT", Fitness, weights=(-1.0, -1.0))
    creator.create_type("PF_IND", list, fitness=creator.__dict__["PF_FIT"])
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
