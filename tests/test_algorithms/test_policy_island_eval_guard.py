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

FIT = "POLICY_ISLAND_GUARD_FIT"
IND = "POLICY_ISLAND_GUARD_IND"


def test_step_islands_guard_charges_pre_action_estimate():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    ind_cls = creator.__dict__[IND]
    evals = {"n": 0}

    def evaluate(individual):
        evals["n"] += 1
        return (float(individual[0]),)

    def vary(population):
        return list(population) + [ind_cls([99.0]) for _ in population]

    toolbox = Toolbox()
    toolbox.register("evaluate", evaluate)
    toolbox.register("vary", vary)
    toolbox.register("select", tools.sel_best)
    population = [ind_cls([1.0]), ind_cls([2.0])]
    guard = tools.PolicyActionGuard(n_evals=4)
    try:
        first = tools.apply_policy_action(
            "step_islands", demes=[(toolbox, population)], guard=guard
        )
        second = tools.apply_policy_action(
            "step_islands", demes=[(toolbox, population)], guard=guard
        )
    finally:
        del creator.__dict__[FIT]
        del creator.__dict__[IND]

    assert first.applied is True
    assert second.rejected is True
    assert guard.nevals_used == 4
    assert evals["n"] == 4


def test_ea_policy_charges_step_islands_pre_action_estimate():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    ind_cls = creator.__dict__[IND]
    evals = {"n": 0}

    def evaluate(individual):
        evals["n"] += 1
        return (float(individual[0]),)

    deme_toolbox = Toolbox()
    deme_toolbox.register("evaluate", evaluate)
    deme_toolbox.register("vary", lambda population: [ind_cls([9.0]) for _ in population])
    deme_toolbox.register("select", tools.sel_best)
    demes = [(deme_toolbox, [ind_cls([1.0]), ind_cls([2.0])]) for _ in range(2)]
    toolbox = Toolbox()
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cx_two_point)
    toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.0)
    toolbox.register("select", tools.sel_best)
    try:
        _, logbook = tools.ea_policy(
            toolbox,
            [ind_cls([0.0])],
            lambda _obs: "step_islands",
            generations=1,
            cx_prob=0.0,
            mut_prob=0.0,
            action_kwargs={"demes": demes},
        )
    finally:
        del creator.__dict__[FIT]
        del creator.__dict__[IND]

    assert evals["n"] == 9
    assert logbook.select("nevals") == [1, 8]
