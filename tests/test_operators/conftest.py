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
from deap_er import Fitness, creator

SO_FIT = "SEL_SO_FIT"
SO_IND = "SEL_SO_IND"
MO_FIT = "SEL_MO_FIT"
MO_IND = "SEL_MO_IND"


@pytest.fixture
def single_obj():
    creator.create_type(SO_FIT, Fitness, weights=(1.0,))
    creator.create_type(SO_IND, list, fitness=creator.__dict__[SO_FIT])
    yield creator.__dict__[SO_IND]
    del creator.__dict__[SO_FIT]
    del creator.__dict__[SO_IND]


@pytest.fixture
def multi_obj():
    creator.create_type(MO_FIT, Fitness, weights=(1.0, 1.0))
    creator.create_type(MO_IND, list, fitness=creator.__dict__[MO_FIT])
    yield creator.__dict__[MO_IND]
    del creator.__dict__[MO_FIT]
    del creator.__dict__[MO_IND]


@pytest.fixture
def make():
    def _make(ind_cls, genes, values):
        ind = ind_cls(genes)
        ind.fitness.values = values
        return ind

    return _make
