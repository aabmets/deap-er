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

from deap_er import Fitness, creator, tools


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


def test_pareto_front_skips_invalid_and_non_finite_fitness():
    creator.create_type("PF_SKIP_FIT", Fitness, weights=(1.0, 1.0))
    creator.create_type("PF_SKIP_IND", list, fitness=creator.__dict__["PF_SKIP_FIT"])
    try:
        front = tools.ParetoFront()
        unevaluated = creator.__dict__["PF_SKIP_IND"]([0])
        member = creator.__dict__["PF_SKIP_IND"]([1])
        member.fitness.values = (1.0, 4.0)
        later = creator.__dict__["PF_SKIP_IND"]([2])
        later.fitness.values = (4.0, 1.0)
        nan_ind = creator.__dict__["PF_SKIP_IND"]([3])
        nan_ind.fitness.values = (math.nan, math.nan)
        front.update([unevaluated, member, nan_ind, later])
        assert len(front) == 2
        assert {ind.fitness.values for ind in front} == {(1.0, 4.0), (4.0, 1.0)}
    finally:
        del creator.__dict__["PF_SKIP_FIT"]
        del creator.__dict__["PF_SKIP_IND"]


def test_pareto_front_skips_individual_without_fitness():
    creator.create_type("PF_FIT", Fitness, weights=(-1.0, -1.0))
    creator.create_type("PF_IND", list, fitness=creator.__dict__["PF_FIT"])
    try:

        class Bare(list[Any]):
            pass

        front = tools.ParetoFront()
        member = creator.__dict__["PF_IND"]([0])
        member.fitness.values = (1.0, 4.0)
        later = creator.__dict__["PF_IND"]([1])
        later.fitness.values = (4.0, 1.0)
        front.update([member])
        front.update([Bare([9]), later])
        assert len(front) == 2
        assert {ind.fitness.values for ind in front} == {(1.0, 4.0), (4.0, 1.0)}
    finally:
        del creator.__dict__["PF_FIT"]
        del creator.__dict__["PF_IND"]
