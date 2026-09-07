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
from deap_er import Fitness, creator, tools

FIT = "MOCMA_SHORT_FIT"
IND = "MOCMA_SHORT_IND"


def test_generate_after_partial_valid_when_lambda_equals_mu():
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        parents = [creator.__dict__[IND]([0.1 * i, 0.2]) for i in range(4)]
        mo = tools.StrategyMultiObjective(parents, sigma=0.1, survivors=4, offsprings=4)
        kids = mo.generate(creator.__dict__[IND])
        for child in kids[:2]:
            child.fitness.values = (float(child[0]), float(child[1]))
        mo.update(kids)
        assert len(mo.parents) == 2
        more = mo.generate(creator.__dict__[IND])
        assert len(more) == mo.lamb
        assert all(child.ps_[0] == "o" for child in more)
    finally:
        del creator.__dict__[FIT]
        del creator.__dict__[IND]
