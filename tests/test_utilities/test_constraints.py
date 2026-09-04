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

FIT = "CON_FIT"
IND = "CON_IND"


def _setup(weights: tuple[float, ...] = (-1.0,)) -> None:
    creator.create(FIT, base.Fitness, weights=weights)
    creator.create(IND, list, fitness=creator.__dict__[FIT])


def _teardown() -> None:
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _ind(genes, values=None):
    individual = creator.__dict__[IND](genes)
    if values is not None:
        individual.fitness.values = values
    else:
        individual.fitness.weights = creator.__dict__[FIT].weights
    return individual


def test_delta_penalty_keeps_valid_and_penalizes_invalid():
    _setup()
    try:
        valid = _ind([0.0])
        invalid = _ind([10.0])
        evaluate = tools.DeltaPenalty(lambda ind: ind[0] < 1.0, 100.0)(lambda ind: (ind[0],))
        assert evaluate(valid) == (0.0,)
        assert evaluate(invalid) == (100.0,)

        with_dist = tools.DeltaPenalty(lambda _ind: False, [5.0], distance=lambda _ind: 2.0)(
            lambda _ind: (0.0,)
        )
        assert with_dist(invalid) == (7.0,)
    finally:
        _teardown()

    _setup((-1.0, 1.0))
    try:
        individual = _ind([1.0])
        penalize = tools.DeltaPenalty(
            lambda _ind: False, [3.0, 4.0], distance=lambda _ind: [1.0, 2.0]
        )(lambda _ind: (0.0, 0.0))
        assert penalize(individual) == (4.0, 2.0)
    finally:
        _teardown()


def test_closest_valid_penalty_and_weight_mismatch():
    _setup()
    try:
        valid = _ind([0.5])
        invalid = _ind([9.0])
        evaluate = tools.ClosestValidPenalty(
            lambda ind: ind[0] < 1.0,
            lambda _ind: valid,
            alpha=2.0,
        )(lambda ind: (ind[0],))
        assert evaluate(valid) == (0.5,)
        assert evaluate(invalid) == (0.5,)

        with_dist = tools.ClosestValidPenalty(
            lambda _ind: False,
            lambda _ind: valid,
            alpha=1.0,
            distance=lambda _feasible, _ind: 3.0,
        )(lambda ind: (ind[0],))
        assert with_dist(invalid) == (3.5,)

        mismatch = tools.ClosestValidPenalty(lambda _ind: False, lambda _ind: valid, alpha=1.0)(
            lambda _ind: (0.0, 1.0)
        )
        with pytest.raises(IndexError, match="Fitness weights"):
            mismatch(invalid)
    finally:
        _teardown()
