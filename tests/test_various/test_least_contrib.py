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
from deap_er import Fitness, creator, tools

FIT = "LC_FIT"
IND = "LC_IND"


def _setup() -> None:
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])


def _teardown() -> None:
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _ind(values: tuple[float, ...]):
    individual = creator.__dict__[IND]()
    individual.fitness.values = values
    return individual


def test_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        tools.least_contrib([])


def test_least_index():
    _setup()
    try:
        pop = [
            _ind((5.0, 5.0)),
            _ind((4.0, 6.0)),
            _ind((2.0, 7.0)),
            _ind((7.0, 4.0)),
        ]
        idx = tools.least_contrib(pop, [10.0, 10.0])
    finally:
        _teardown()
    assert idx == 1
