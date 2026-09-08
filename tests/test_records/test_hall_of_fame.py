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
import json
import math
from operator import eq
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


def test_update_skips_invalid_fitness_and_keeps_later_members(ind_cls):
    unevaluated = ind_cls([0])
    good = ind_cls([9])
    good.fitness.values = (7.0,)
    hof = tools.HallOfFame(maxsize=5)
    hof.update([unevaluated, good])
    assert len(hof) == 1
    assert list(hof[0]) == [9]
    assert hof[0].fitness.values == (7.0,)


def test_update_rejects_non_finite_fitness(ind_cls):
    finite = ind_cls([1])
    finite.fitness.values = (10.0,)
    nan_ind = ind_cls([2])
    nan_ind.fitness.values = (math.nan,)
    hof = tools.HallOfFame(maxsize=2)
    hof.update([finite, nan_ind])
    assert len(hof) == 1
    assert list(hof[0]) == [1]
    assert hof[0].fitness.values == (10.0,)

    empty = tools.HallOfFame(maxsize=2)
    empty.update([nan_ind])
    assert len(empty) == 0
    inf_ind = ind_cls([3])
    inf_ind.fitness.values = (math.inf,)
    empty.update([inf_ind])
    assert len(empty) == 0


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


def test_json_round_trip_restores_members(ind_cls):
    hof = tools.HallOfFame(maxsize=3)
    hof.update(_population(ind_cls))

    restored = tools.HallOfFame.from_json(hof.to_json(), ind_cls)

    assert restored.maxsize == 3
    assert [list(ind) for ind in restored] == [list(ind) for ind in hof]
    assert [ind.fitness.values for ind in restored] == [ind.fitness.values for ind in hof]


def test_json_round_trip_empty_archive(ind_cls):
    hof = tools.HallOfFame(maxsize=2)
    restored = tools.HallOfFame.from_json(hof.to_json(), ind_cls)
    assert restored.maxsize == 2
    assert len(restored) == 0


def test_from_json_requires_ind_cls_for_members(ind_cls):
    hof = tools.HallOfFame(maxsize=1)
    hof.update(_population(ind_cls, count=1))
    with pytest.raises(ValueError, match="ind_cls"):
        tools.HallOfFame.from_json(hof.to_json())


def test_json_round_trip_resets_similar_to_default_eq(ind_cls):
    def always_similar(_left: Any, _right: Any) -> bool:
        return True

    hof = tools.HallOfFame(maxsize=2, similar=always_similar)
    first = ind_cls([1])
    first.fitness.values = (10.0,)
    second = ind_cls([2])
    second.fitness.values = (5.0,)
    hof.update([first, second])
    assert hof.similar is always_similar
    assert len(hof) == 1

    payload = json.loads(hof.to_json())
    assert "similar" not in payload

    restored = tools.HallOfFame.from_json(hof.to_json(), ind_cls)
    assert restored.similar is eq
    third = ind_cls([3])
    third.fitness.values = (1.0,)
    restored.update([third])
    assert len(restored) == 2


def test_json_round_trip_drops_extra_individual_attributes(ind_cls):
    hof = tools.HallOfFame(maxsize=1)
    ind = ind_cls([42])
    ind.fitness.values = (9.0,)
    ind.extra_tag = "survivor"  # type: ignore[attr-defined]
    hof.update([ind])

    payload = json.loads(hof.to_json())
    assert set(payload["items"][0]) == {"genes", "fitness"}

    restored = tools.HallOfFame.from_json(hof.to_json(), ind_cls)
    assert list(restored[0]) == [42]
    assert restored[0].fitness.values == (9.0,)
    assert not hasattr(restored[0], "extra_tag")
