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

SO_FIT = "CMA_STD_FIT"
SO_IND = "CMA_STD_IND"


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
        del creator.__dict__[SO_FIT]
        del creator.__dict__[SO_IND]


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
        del creator.__dict__[SO_FIT]
        del creator.__dict__[SO_IND]
