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
from deap_er import Fitness, Toolbox, creator, tools

FIT = "EA_GU_FIT"
IND = "EA_GU_IND"


def test_standard_cma():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        dimensions = 5
        strategy = tools.Strategy(centroid=[0.0] * dimensions, sigma=1.0)

        toolbox = Toolbox()
        toolbox.register("evaluate", tools.bm_sphere)
        toolbox.register("generate", strategy.generate, creator.__dict__[IND])
        toolbox.register("update", strategy.update)

        pop, _ = tools.ea_generate_update(toolbox, generations=100)
        (best,) = tools.sel_best(pop, sel_count=1)

        assert best.fitness.values < (1e-8,)
    finally:
        del creator.__dict__[FIT]
        del creator.__dict__[IND]
