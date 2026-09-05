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

CD_FIT = "CD_FIT"
CD_IND = "CD_IND"


def test_assign_crowding_dist_use_weights_changes_order():
    creator.create_type(CD_FIT, Fitness, weights=(1.0, 0.0))
    creator.create_type(CD_IND, list, fitness=creator.__dict__[CD_FIT])
    try:
        ind_cls = creator.__dict__[CD_IND]

        def _pop():
            people = []
            for genes, values in (([0], (0.0, 0.0)), ([1], (1.0, 1.0)), ([2], (2.0, 2.0))):
                individual = ind_cls(genes)
                individual.fitness.values = values
                people.append(individual)
            return people

        by_values = _pop()
        by_weights = _pop()
        tools.assign_crowding_dist(by_values)
        tools.assign_crowding_dist(by_weights, use_weights=True)
        left = [ind.fitness.crowding_dist for ind in by_values]
        right = [ind.fitness.crowding_dist for ind in by_weights]
        assert left != right
    finally:
        del creator.__dict__[CD_FIT]
        del creator.__dict__[CD_IND]
