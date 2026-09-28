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

REUSE_FIT = "ISLAND_REUSE_FIT"
REUSE_IND = "ISLAND_REUSE_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(REUSE_FIT, Fitness, weights=(1.0,))
    creator.create_type(REUSE_IND, list, fitness=creator.__dict__[REUSE_FIT])
    yield creator.__dict__[REUSE_IND]
    del creator.__dict__[REUSE_FIT]
    del creator.__dict__[REUSE_IND]


def test_step_islands_invalidates_an_immigrant_that_reuses_a_freed_id(ind_cls):
    toolbox = Toolbox()
    toolbox.register("evaluate", lambda individual: (float(individual[0]),))
    toolbox.register("vary", list)
    toolbox.register("select", tools.sel_best)

    def migrate(populations):
        source = populations[1][0]
        populations[0].clear()
        immigrant = ind_cls(source)
        immigrant.fitness.values = source.fitness.values
        populations[0].append(immigrant)

    for _ in range(50):
        first = [ind_cls([1.0])]
        second = [ind_cls([2.0])]
        tools.step_islands([(toolbox, first), (toolbox, second)], migrate, eval_keys=("a", "b"))
        assert not first[0].fitness.is_valid()
