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
import pytest
from deap_er import Fitness, creator, tools
from deap_er.private.various.rng import rng

FIT = "CMA_COLLAPSE_FIT"
IND = "CMA_COLLAPSE_IND"
STATE = ("centroid", "sigma", "ps", "pc", "big_c", "diag_d", "update_count")
CLASSES = [tools.Strategy, tools.StrategySeparable]


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    rng.seed(0)
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _corner_strategy(cls, sigma):
    """Clip-bounded ``sum(x)`` minimum sits on the corner, so sigma only shrinks."""
    return cls([0.0] * 4, sigma, offsprings=8, low=0.0, up=1.0)


def _step(strategy, ind_cls):
    population = strategy.generate(ind_cls)
    for individual in population:
        individual.fitness.values = (sum(individual),)
    strategy.update(population)


@pytest.mark.parametrize("cls", CLASSES)
def test_update_stays_finite_after_sigma_squared_underflows(cls, ind_cls):
    strategy = _corner_strategy(cls, 1e-170)
    assert strategy.sigma**2 == 0.0
    for _ in range(5):
        _step(strategy, ind_cls)
    assert strategy.update_count == 5
    assert numpy.isfinite(strategy.big_c).all()
    assert numpy.isfinite(strategy.diag_d).all()
    assert 0.0 < strategy.sigma < 1e-170


@pytest.mark.parametrize("cls", CLASSES)
def test_collapsed_sigma_raises_and_leaves_state_unchanged(cls, ind_cls):
    strategy = _corner_strategy(cls, 1e-300)
    for _ in range(2000):
        before = {name: numpy.copy(getattr(strategy, name)) for name in STATE}
        try:
            _step(strategy, ind_cls)
        except FloatingPointError:
            break
    else:
        pytest.fail("sigma never collapsed")
    for name, value in before.items():
        numpy.testing.assert_array_equal(getattr(strategy, name), value, err_msg=name)


def _evaluated_strategy(ind_cls):
    strategy = tools.Strategy([0.0, 0.0], 1.0, offsprings=8)
    population = strategy.generate(ind_cls)
    for individual in population:
        individual.fitness.values = (sum(x * x for x in individual),)
    return strategy, population


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_update_rolls_back_when_the_covariance_overflows_its_decomposition(ind_cls):
    strategy, population = _evaluated_strategy(ind_cls)
    strategy.big_c = numpy.full((2, 2), 1.7e308)
    before = {name: numpy.copy(getattr(strategy, name)) for name in (*STATE, "big_b", "cond")}

    with pytest.raises(FloatingPointError):
        strategy.update(population)
    for name, value in before.items():
        numpy.testing.assert_array_equal(getattr(strategy, name), value, err_msg=name)
    assert numpy.isfinite(strategy.generate(ind_cls)).all()


def test_update_rolls_back_when_eigh_does_not_converge(ind_cls, monkeypatch):
    strategy, population = _evaluated_strategy(ind_cls)
    before = {name: numpy.copy(getattr(strategy, name)) for name in STATE}

    def no_convergence(_matrix):
        raise numpy.linalg.LinAlgError("Eigenvalues did not converge")

    monkeypatch.setattr(numpy.linalg, "eigh", no_convergence)
    with pytest.raises(FloatingPointError, match="did not converge"):
        strategy.update(population)
    for name, value in before.items():
        numpy.testing.assert_array_equal(getattr(strategy, name), value, err_msg=name)


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_restart_strategy_restarts_after_a_failed_decomposition(ind_cls):
    strategy = tools.Strategy([0.0, 0.0], 1.0, offsprings=8)
    restart = tools.RestartStrategy(strategy, mode="ipop", budget=1000)
    population = restart.generate(ind_cls)
    for individual in population:
        individual.fitness.values = (sum(x * x for x in individual),)
    strategy.big_c = numpy.full((2, 2), 1.7e308)
    restart.update(population)
    assert strategy.update_count == 0
    assert restart.should_restart()


def test_restart_strategy_restarts_a_collapsed_run(ind_cls):
    strategy = tools.Strategy([0.5] * 4, 1.0, offsprings=8, low=0.0, up=1.0)
    restart = tools.RestartStrategy(strategy, mode="ipop", budget=1000)
    population = restart.generate(ind_cls)
    for individual in population:
        individual.fitness.values = (sum(individual),)
    strategy.sigma = 0.0
    restart.update(population)
    assert restart.evals_used == 8
    assert strategy.update_count == 0
    assert restart.should_restart()
    restart.restart()
    assert strategy.sigma > 0.0
