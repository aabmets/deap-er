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
import numpy
from deap_er import Fitness, creator, tools

SO_FIT = "CMA_BND_FIT"
SO_IND = "CMA_BND_IND"
MO_FIT = "CMA_BND_MO_FIT"
MO_IND = "CMA_BND_MO_IND"


def _teardown(*names: str) -> None:
    for name in names:
        del creator.__dict__[name]


def test_strategy_clip_keeps_genes_in_box():
    creator.create_type(SO_FIT, Fitness, weights=(-1.0,))
    creator.create_type(SO_IND, list, fitness=creator.__dict__[SO_FIT])
    try:
        strategy = tools.Strategy(centroid=[5.0, 5.0], sigma=3.0, low=0.0, up=1.0)
        tools.rng.seed(1)
        population = strategy.generate(creator.__dict__[SO_IND])
        assert len(population) == strategy.lamb
        for individual in population:
            assert all(0.0 <= gene <= 1.0 for gene in individual)
    finally:
        _teardown(SO_FIT, SO_IND)


def test_strategy_resample_falls_back_and_compute_params_keeps_bounds():
    creator.create_type(SO_FIT, Fitness, weights=(-1.0,))
    creator.create_type(SO_IND, list, fitness=creator.__dict__[SO_FIT])
    try:
        strategy = tools.Strategy(
            centroid=[10.0, 10.0],
            sigma=2.0,
            low=0.0,
            up=0.1,
            bound_mode="resample",
            resample_limit=2,
        )
        tools.rng.seed(2)
        population = strategy.generate(creator.__dict__[SO_IND])
        assert len(population) == strategy.lamb
        for individual in population:
            assert all(0.0 <= gene <= 0.1 for gene in individual)

        strategy.compute_params()
        assert strategy.low == 0.0
        assert strategy.up == 0.1
        assert strategy.bound_mode == "resample"
    finally:
        _teardown(SO_FIT, SO_IND)


def test_multi_objective_generate_respects_box_bounds():
    creator.create_type(MO_FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(MO_IND, numpy.ndarray, fitness=creator.__dict__[MO_FIT])
    try:
        tools.rng.seed(3)
        parents = [creator.__dict__[MO_IND]([0.2, 0.3, 0.4]) for _ in range(4)]
        for parent in parents:
            parent.fitness.values = tools.bm_zdt_1(parent)
        strategy = tools.StrategyMultiObjective(
            parents, sigma=8.0, survivors=4, offsprings=4, low=0.0, up=1.0
        )
        children = strategy.generate(creator.__dict__[MO_IND])
        assert len(children) == 4
        for child in children:
            genes = numpy.asarray(child)
            assert numpy.all(genes >= 0.0) and numpy.all(genes <= 1.0)
    finally:
        _teardown(MO_FIT, MO_IND)
