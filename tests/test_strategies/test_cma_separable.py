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
from deap_er import Fitness, Toolbox, creator, tools
from deap_er.private.strategies.restart_common import strategy_diagnostics

SO_FIT = "CMA_SEP_FIT"
SO_IND = "CMA_SEP_IND"


def _types():
    creator.create_type(SO_FIT, Fitness, weights=(-1.0,))
    creator.create_type(SO_IND, list, fitness=creator.__dict__[SO_FIT])
    return creator.__dict__[SO_IND]


def _teardown():
    del creator.__dict__[SO_FIT]
    del creator.__dict__[SO_IND]


def test_separable_clip_keeps_genes_in_box():
    ind_cls = _types()
    try:
        strategy = tools.StrategySeparable(centroid=[5.0, 5.0], sigma=3.0, low=0.0, up=1.0)
        tools.rng.seed(1)
        population = strategy.generate(ind_cls)
        assert len(population) == strategy.lamb
        for individual in population:
            assert all(0.0 <= gene <= 1.0 for gene in individual)
    finally:
        _teardown()


def test_separable_resample_and_compute_params_keeps_bounds():
    ind_cls = _types()
    try:
        strategy = tools.StrategySeparable(
            centroid=[10.0, 10.0],
            sigma=2.0,
            low=0.0,
            up=0.1,
            bound_mode="resample",
            resample_limit=2,
        )
        tools.rng.seed(2)
        population = strategy.generate(ind_cls)
        assert len(population) == strategy.lamb
        for individual in population:
            assert all(0.0 <= gene <= 0.1 for gene in individual)
        strategy.compute_params()
        assert strategy.low == 0.0
        assert strategy.up == 0.1
        assert strategy.bound_mode == "resample"
    finally:
        _teardown()


def test_separable_keeps_learned_c_unless_cm_init():
    ind_cls = _types()
    try:
        strategy = tools.StrategySeparable([0.0, 0.0], 1.0)
        assert strategy.big_c.shape == (2,)
        assert numpy.allclose(strategy.big_c, numpy.ones(2))
        tools.rng.seed(0)
        for _ in range(8):
            population = strategy.generate(ind_cls)
            for individual in population:
                individual.fitness.values = (sum(gene * gene for gene in individual),)
            strategy.update(population)
        learned_c = numpy.array(strategy.big_c, copy=True)
        learned_pc = numpy.array(strategy.pc, copy=True)
        learned_count = strategy.update_count
        assert learned_c.ndim == 1
        assert not numpy.allclose(learned_c, numpy.ones(2))
        strategy.compute_params(offsprings=12)
        assert numpy.allclose(strategy.big_c, learned_c)
        assert numpy.allclose(strategy.pc, learned_pc)
        assert strategy.update_count == learned_count
        assert strategy.lamb == 12
        strategy.compute_params(cm_init=numpy.array([2.0, 0.5]))
        assert numpy.allclose(strategy.big_c, [2.0, 0.5])
    finally:
        _teardown()


def test_separable_weight_schemes_cm_init_and_bounds():
    strategy = tools.StrategySeparable([0.0, 0.0], 1.0, offsprings=6, survivors=3)
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
    scalar = tools.StrategySeparable([0.0, 0.0], 1.0, cm_init=2.0)
    assert numpy.allclose(scalar.big_c, [2.0, 2.0])
    try:
        tools.StrategySeparable([0.0, 0.0], 1.0, cm_init=numpy.identity(2))
    except ValueError as err:
        assert "cm_init" in str(err)
    else:
        raise AssertionError("expected ValueError for 2-D cm_init")
    try:
        tools.StrategySeparable([0.0, 0.0], 1.0, bound_mode="wrap")
    except ValueError as err:
        assert "bound_mode" in str(err)
    else:
        raise AssertionError("expected ValueError for bound_mode")
    try:
        tools.StrategySeparable([0.0, 0.0], 1.0, resample_limit=0)
    except ValueError as err:
        assert "resample_limit" in str(err)
    else:
        raise AssertionError("expected ValueError for resample_limit")


def test_separable_learning_rates_scale_and_stay_one_d():
    dim = 10
    standard = tools.Strategy([0.0] * dim, 1.0)
    separable = tools.StrategySeparable([0.0] * dim, 1.0)
    scale = (dim + 2.0) / 3.0
    expected_one = 2.0 / ((dim + 1.3) ** 2 + standard.mu_eff) * scale
    expected_mu = 2.0 * (standard.mu_eff - 2.0 + 1.0 / standard.mu_eff)
    expected_mu /= (dim + 2.0) ** 2 + standard.mu_eff
    expected_mu = min(1 - expected_one, expected_mu * scale)
    assert abs(separable.rank_one - expected_one) < 1e-12
    assert abs(separable.rank_mu - expected_mu) < 1e-12
    explicit = tools.StrategySeparable([0.0] * dim, 1.0, rank_one=0.01, rank_mu=0.02)
    assert explicit.rank_one == 0.01
    assert abs(explicit.rank_mu - min(1 - 0.01, 0.02)) < 1e-12
    assert not hasattr(separable, "big_b")
    assert not hasattr(separable, "big_bd")


def test_separable_high_n_update_stays_one_d():
    ind_cls = _types()
    try:
        dim = 64
        strategy = tools.StrategySeparable([0.0] * dim, 1.0, offsprings=8, survivors=4)
        tools.rng.seed(3)
        population = strategy.generate(ind_cls)
        for individual in population:
            individual.fitness.values = tools.bm_sphere(individual)
        strategy.update(population)
        assert strategy.big_c.shape == (dim,)
        assert strategy.diag_d.shape == (dim,)
        assert strategy.update_count == 1
    finally:
        _teardown()


def test_separable_reset_state_and_ipop_restart():
    ind_cls = _types()
    try:
        strategy = tools.StrategySeparable([0.0] * 5, 1.0, offsprings=8, low=-5.0, up=5.0)
        tools.rng.seed(4)
        population = strategy.generate(ind_cls)
        for individual in population:
            individual.fitness.values = tools.bm_sphere(individual)
        strategy.update(population)
        assert strategy.update_count == 1
        strategy.reset_state([1.0] * 5, 0.5, offsprings=20)
        assert strategy.update_count == 0
        assert strategy.sigma == 0.5
        assert strategy.lamb == 20
        assert numpy.allclose(strategy.big_c, numpy.ones(5))
        assert numpy.allclose(strategy.pc, 0.0)
        restart = tools.RestartStrategy(strategy, mode="ipop", budget=1_000_000)
        restart.generate(ind_cls)
        restart._tracker.terminate = True
        assert restart.should_restart()
        restart.restart()
        assert strategy.lamb == 40
        assert strategy.update_count == 0
        assert restart.restart_count == 1
    finally:
        _teardown()


def test_separable_diagnostics_use_max_variance():
    strategy = tools.StrategySeparable([0.0] * 4, 0.3, cm_init=[4.0, 1.0, 9.0, 0.25])
    cond, sigma, largest = strategy_diagnostics(strategy)
    assert sigma == 0.3
    assert cond == 9.0 / 0.25
    assert largest == 9.0


def test_separable_sphere_smoke():
    ind_cls = _types()
    try:
        strategy = tools.StrategySeparable([1.0] * 5, 1.0, offsprings=8)
        toolbox = Toolbox()
        toolbox.register("evaluate", tools.bm_sphere)
        toolbox.register("generate", strategy.generate, ind_cls)
        toolbox.register("update", strategy.update)
        tools.ea_generate_update(toolbox, generations=6, verbose=False)
        assert strategy.update_count == 6
        assert numpy.isfinite(strategy.centroid).all()
        assert strategy.big_c.ndim == 1
    finally:
        _teardown()


def test_separable_rejects_non_positive_cm_init():
    cases = (0.0, -1.0, [1.0, 0.0], [1.0, -0.5])
    for cm_init in cases:
        try:
            tools.StrategySeparable([0.0, 0.0], 1.0, cm_init=cm_init)
        except ValueError as err:
            assert "cm_init" in str(err)
        else:
            raise AssertionError(f"expected ValueError for cm_init={cm_init!r}")
    strategy = tools.StrategySeparable([0.0, 0.0], 1.0)
    try:
        strategy.compute_params(cm_init=[2.0, 0.0])
    except ValueError as err:
        assert "cm_init" in str(err)
    else:
        raise AssertionError("expected ValueError for zero variance in compute_params")
