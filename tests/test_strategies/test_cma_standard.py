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


def test_compute_params_keeps_learned_c_unless_cm_init():
    creator.create_type(SO_FIT, Fitness, weights=(-1.0,))
    creator.create_type(SO_IND, list, fitness=creator.__dict__[SO_FIT])
    try:
        strategy = tools.Strategy([0.0, 0.0], 1.0)
        assert numpy.allclose(strategy.big_c, numpy.identity(2))
        tools.rng.seed(0)
        for _ in range(8):
            population = strategy.generate(creator.__dict__[SO_IND])
            for individual in population:
                individual.fitness.values = (sum(gene * gene for gene in individual),)
            strategy.update(population)

        learned_c = numpy.array(strategy.big_c, copy=True)
        learned_pc = numpy.array(strategy.pc, copy=True)
        learned_count = strategy.update_count
        assert not numpy.allclose(learned_c, numpy.identity(2))

        strategy.compute_params(offsprings=12)
        assert numpy.allclose(strategy.big_c, learned_c)
        assert numpy.allclose(strategy.pc, learned_pc)
        assert strategy.update_count == learned_count
        assert strategy.lamb == 12

        custom = numpy.array([[2.0, 0.0], [0.0, 0.5]])
        strategy.compute_params(cm_init=custom)
        assert numpy.allclose(strategy.big_c, custom)
    finally:
        del creator.__dict__[SO_FIT]
        del creator.__dict__[SO_IND]


def test_compute_params_weight_schemes_and_unknown():
    strategy = tools.Strategy([0.0, 0.0], 1.0, offsprings=6, survivors=3)
    strategy.compute_params(weights="linear")
    assert strategy.weights.shape == (3,)
    strategy.compute_params(weights="equal")
    assert numpy.allclose(strategy.weights, numpy.full(3, 1.0 / 3.0))
    try:
        strategy.compute_params(weights="cubic")
    except RuntimeError as err:
        assert "Unknown weights" in str(err)
    else:
        raise AssertionError("expected RuntimeError for unknown weights")


def test_invalid_bound_mode_and_resample_limit():
    try:
        tools.Strategy([0.0, 0.0], 1.0, bound_mode="wrap")
    except ValueError as err:
        assert "bound_mode" in str(err)
    else:
        raise AssertionError("expected ValueError for bound_mode")
    try:
        tools.Strategy([0.0, 0.0], 1.0, resample_limit=0)
    except ValueError as err:
        assert "resample_limit" in str(err)
    else:
        raise AssertionError("expected ValueError for resample_limit")
