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
from deap_er.private.operators.sel_moead import _update_ideal_point


def test_tchebycheff_known_values():
    fitness = numpy.array([[1.0, 4.0], [3.0, 2.0]])
    weights = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    ideal = numpy.array([0.0, 0.0])
    got = tools.scalarization_tchebycheff(fitness, weights, ideal)
    assert got[0, 0] == pytest.approx(1.0)
    assert got[0, 1] == pytest.approx(4.0)
    assert got[1, 0] == pytest.approx(3.0)
    assert got[1, 1] == pytest.approx(2.0)


def test_pbi_known_values():
    fitness = numpy.array([[1.0, 1.0]])
    weights = numpy.array([[1.0, 1.0]])
    ideal = numpy.array([0.0, 0.0])
    got = tools.scalarization_pbi(fitness, weights, ideal, theta=5.0)
    d1 = numpy.sqrt(2.0)
    assert got[0, 0] == pytest.approx(d1)


def test_zero_weight_eps():
    fitness = numpy.array([[1.0, 2.0]])
    weights = numpy.array([[0.0, 1.0]])
    ideal = numpy.array([0.0, 0.0])
    got = tools.scalarization_tchebycheff(fitness, weights, ideal)
    assert numpy.isfinite(got).all()


def test_ideal_point_memory_updates(multi_obj, make):
    weights = tools.uniform_reference_points(2, 4)
    select = tools.SelMOEADWithMemory(weights)
    tools.rng.seed(3)
    for _ in range(2):
        pop = [
            make(multi_obj, [tools.rng.random()], (tools.rng.random(), tools.rng.random()))
            for _ in range(10)
        ]
        select(pop, 4)
    assert numpy.all(numpy.isfinite(select.ideal_point))
    assert numpy.all(select.ideal_point <= 1.0)


def test_moead_memory_updates_on_full_pool_select():
    fit_name = "MOEAD_SCALAR_FULL_FIT"
    ind_name = "MOEAD_SCALAR_FULL_IND"
    creator.create_type(fit_name, Fitness, weights=(-1.0, -1.0))
    creator.create_type(ind_name, list, fitness=creator.__dict__[fit_name])
    try:
        weights = tools.uniform_reference_points(2, 4)
        select = tools.SelMOEADWithMemory(weights)
        pop = []
        for genes, values in (([0.0], (0.01, 0.02)), ([1.0], (0.2, 0.8)), ([2.0], (0.8, 0.2))):
            ind = creator.__dict__[ind_name](genes)
            ind.fitness.values = values
            pop.append(ind)

        select(pop, len(pop))
        assert numpy.all(numpy.isfinite(select.ideal_point))
        assert select.ideal_point.reshape(-1)[0] == pytest.approx(0.01)
        assert select.ideal_point.reshape(-1)[1] == pytest.approx(0.02)
    finally:
        del creator.__dict__[fit_name]
        del creator.__dict__[ind_name]


def test_update_ideal_point_mixed_inf():
    fitness = numpy.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    got = _update_ideal_point(fitness, numpy.array([0.1, numpy.inf, 0.2]))
    assert got.shape == (3,)
    assert got.tolist() == pytest.approx([0.1, 1.5, 0.2])


def test_update_ideal_point_raises_on_shape_mismatch():
    fitness = numpy.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="objective count"):
        _update_ideal_point(fitness, numpy.array([0.5]))


def test_sel_moead_raises_on_ideal_point_shape_mismatch():
    fit_name = "MOEAD_SCALAR_SHAPE_FIT"
    ind_name = "MOEAD_SCALAR_SHAPE_IND"
    creator.create_type(fit_name, Fitness, weights=(-1.0, -1.0))
    creator.create_type(ind_name, list, fitness=creator.__dict__[fit_name])
    try:
        pop = []
        for genes, values in (([0.0], (0.2, 0.8)), ([1.0], (0.8, 0.2))):
            ind = creator.__dict__[ind_name](genes)
            ind.fitness.values = values
            pop.append(ind)

        weights = numpy.array([[0.5, 0.5]])
        with pytest.raises(ValueError, match="objective count"):
            tools.sel_moead(pop, 1, weights, ideal_point=numpy.array([0.0]))
    finally:
        del creator.__dict__[fit_name]
        del creator.__dict__[ind_name]


def test_sel_moead_accepts_mixed_inf_ideal_point():
    fit_name = "MOEAD_SCALAR_INF_FIT"
    ind_name = "MOEAD_SCALAR_INF_IND"
    creator.create_type(fit_name, Fitness, weights=(-1.0, -1.0))
    creator.create_type(ind_name, list, fitness=creator.__dict__[fit_name])
    try:
        pop = []
        for genes, values in (([0.0], (0.2, 0.8)), ([1.0], (0.8, 0.2))):
            ind = creator.__dict__[ind_name](genes)
            ind.fitness.values = values
            pop.append(ind)

        weights = numpy.array([[0.5, 0.5]])
        prior = numpy.array([0.0, numpy.inf])
        chosen = tools.sel_moead(pop, 1, weights, ideal_point=prior)
        assert len(chosen) == 1
    finally:
        del creator.__dict__[fit_name]
        del creator.__dict__[ind_name]
