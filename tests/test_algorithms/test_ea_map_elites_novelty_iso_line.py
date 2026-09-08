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

ME_NOV_FIT = "ME_NOV_FIT"
ME_NOV_IND = "ME_NOV_IND"

LOW, UP = 0.0, 1.0
DIM = 4


def _descriptor(individual):
    return (float(individual[0]), float(individual[1]))


def _evaluate(individual):
    return (sum(individual),)


def _make_toolbox(archive):
    creator.create_type(ME_NOV_FIT, Fitness, weights=(1.0,))
    creator.create_type(ME_NOV_IND, list, fitness=creator.__dict__[ME_NOV_FIT])

    tb = Toolbox()
    tb.register("attr_float", tools.rng.uniform, LOW, UP)
    tb.register(
        "individual",
        tools.init_repeat,
        creator.__dict__[ME_NOV_IND],
        tb.attr_float,
        DIM,
    )
    tb.register("population", tools.init_repeat, list, tb.individual)
    tb.register("mate", tools.cx_blend_bounded, alpha=0.5, low=LOW, up=UP)
    tb.register("evaluate", _evaluate)

    def mutate(individual):
        donor = archive.random_elites(1)[0]
        return tools.mut_iso_line(individual, donor, iso=0.05, sigma=0.02, low=LOW, up=UP)

    tb.register("mutate", mutate)
    tb.register(
        "select",
        tools.sel_novelty,
        archive=archive,
        descriptor_fn=_descriptor,
        k=2,
    )
    return tb


def _population(toolbox, count=16):
    tools.rng.seed(11)
    return toolbox.population(count)


def test_ea_map_elites_runs_with_mut_iso_line_and_sel_novelty_registration():
    archive = tools.UnstructuredArchive(2, min_distance=0.15, max_elites=12)
    toolbox = _make_toolbox(archive)
    initial = _population(toolbox)

    result, logbook = tools.ea_map_elites(
        toolbox,
        archive,
        _descriptor,
        initial,
        generations=3,
        batch_size=10,
        cx_prob=0.3,
        mut_prob=0.7,
    )

    assert len(result) >= 2
    assert logbook.select("gen") == [0, 1, 2, 3]

    near = toolbox.individual()
    near[0], near[1] = 0.01, 0.02
    near[2], near[3] = 0.5, 0.5
    far = toolbox.individual()
    far[0], far[1] = 0.95, 0.90
    far[2], far[3] = 0.5, 0.5
    near.fitness.values = (1.0,)
    far.fitness.values = (1.0,)

    selected = toolbox.select([near, far], 1)
    assert selected == [far]

    del creator.__dict__[ME_NOV_FIT]
    del creator.__dict__[ME_NOV_IND]
