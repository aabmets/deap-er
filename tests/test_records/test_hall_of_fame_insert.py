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

from deap_er import Fitness, creator, tools

HOF_FIT = "HOF_INS_FIT"
HOF_IND = "HOF_INS_IND"
PF_FIT = "PF_INS_FIT"
PF_IND = "PF_INS_IND"


def test_insert_skips_invalid_and_non_finite_fitness():
    creator.create_type(HOF_FIT, Fitness, weights=(1.0,))
    creator.create_type(HOF_IND, list, fitness=creator.__dict__[HOF_FIT])
    try:
        hof = tools.HallOfFame(5)
        unevaluated = creator.__dict__[HOF_IND]([0])
        nan_ind = creator.__dict__[HOF_IND]([1])
        nan_ind.fitness.values = (math.nan,)
        valid = creator.__dict__[HOF_IND]([2])
        valid.fitness.values = (4.0,)
        hof.insert(unevaluated)
        hof.insert(nan_ind)
        hof.insert(valid)
        assert len(hof) == 1
        assert hof[0].fitness.values == (4.0,)
    finally:
        del creator.__dict__[HOF_FIT]
        del creator.__dict__[HOF_IND]


def test_pareto_insert_skips_invalid_and_non_finite_fitness():
    creator.create_type(PF_FIT, Fitness, weights=(1.0, 1.0))
    creator.create_type(PF_IND, list, fitness=creator.__dict__[PF_FIT])
    try:
        front = tools.ParetoFront()
        unevaluated = creator.__dict__[PF_IND]([0])
        nan_ind = creator.__dict__[PF_IND]([1])
        nan_ind.fitness.values = (math.nan, math.nan)
        valid = creator.__dict__[PF_IND]([2])
        valid.fitness.values = (1.0, 4.0)
        front.insert(unevaluated)
        front.insert(nan_ind)
        front.insert(valid)
        assert len(front) == 1
        assert front[0].fitness.values == (1.0, 4.0)
    finally:
        del creator.__dict__[PF_FIT]
        del creator.__dict__[PF_IND]


def test_from_json_does_not_restore_non_finite_fitness():
    creator.create_type(HOF_FIT, Fitness, weights=(1.0,))
    creator.create_type(HOF_IND, list, fitness=creator.__dict__[HOF_FIT])
    try:
        payload = '{"maxsize": 5, "items": [{"genes": [0], "fitness": [NaN]}]}'
        restored = tools.HallOfFame.from_json(payload, creator.__dict__[HOF_IND])
        assert len(restored) == 0
    finally:
        del creator.__dict__[HOF_FIT]
        del creator.__dict__[HOF_IND]
