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

FIT = "SMS_FIT"
IND = "SMS_IND"
HV_THRESHOLD = 116.0

CRITICAL_VALUES = [
    (9.0, 1.0),
    (8.0, 2.0),
    (7.0, 3.0),
    (6.0, 4.0),
    (5.0, 5.0),
    (1.0, 9.0),
    (2.0, 8.0),
    (3.0, 7.0),
    (4.0, 6.0),
]


def _critical_population(multi_obj, make):
    return [make(multi_obj, [index], value) for index, value in enumerate(CRITICAL_VALUES)]


def test_empty_population_returns_empty():
    assert tools.sel_sms_emoa([], 3) == []


def test_sel_count_zero_returns_empty(multi_obj, make):
    population = _critical_population(multi_obj, make)
    assert tools.sel_sms_emoa(population, 0) == []


def test_sel_count_negative_returns_empty(multi_obj, make):
    population = _critical_population(multi_obj, make)
    assert tools.sel_sms_emoa(population, -1) == []


def test_returns_requested_count(multi_obj, make):
    population = _critical_population(multi_obj, make)
    chosen = tools.sel_sms_emoa(population, 4)
    assert len(chosen) == 4


def test_sel_count_exceeds_pool(multi_obj, make):
    population = _critical_population(multi_obj, make)
    chosen = tools.sel_sms_emoa(population, 20)
    assert len(chosen) == len(population)


def test_no_truncation_when_fronts_fit_exactly(multi_obj, make):
    population = [
        make(multi_obj, [0], (5.0, 5.0)),
        make(multi_obj, [1], (4.0, 6.0)),
        make(multi_obj, [2], (6.0, 4.0)),
        make(multi_obj, [3], (1.0, 1.0)),
        make(multi_obj, [4], (2.0, 2.0)),
    ]
    chosen = tools.sel_sms_emoa(population, 3)
    assert [ind[0] for ind in chosen] == [0, 1, 2]


def test_differs_from_nsga2_on_critical_front(multi_obj, make):
    population = _critical_population(multi_obj, make)
    nsga = tools.sel_nsga_2(population, 3)
    sms = tools.sel_sms_emoa(population, 3)
    assert [ind[0] for ind in nsga] == [0, 5, 1]
    assert [ind[0] for ind in sms] == [3, 6, 8]


def test_explicit_ref_point(multi_obj, make):
    population = _critical_population(multi_obj, make)
    default = tools.sel_sms_emoa(population, 3)
    explicit = tools.sel_sms_emoa(population, 3, ref_point=[10.0, 10.0])
    assert [ind[0] for ind in default] == [3, 6, 8]
    assert [ind[0] for ind in explicit] == [0, 4, 5]


def test_critical_front_size_one(multi_obj, make):
    population = [
        make(multi_obj, [0], (5.0, 5.0)),
        make(multi_obj, [1], (4.0, 6.0)),
        make(multi_obj, [2], (6.0, 4.0)),
        make(multi_obj, [3], (4.5, 4.5)),
    ]
    chosen = tools.sel_sms_emoa(population, 4)
    assert [ind[0] for ind in chosen] == [0, 1, 2, 3]


def test_contribution_tie_first_index_wins(multi_obj, make):
    population = [
        make(multi_obj, [0], (4.0, 4.0)),
        make(multi_obj, [1], (4.0, 4.0)),
        make(multi_obj, [2], (5.0, 5.0)),
    ]
    ref = [10.0, 10.0]
    chosen = tools.sel_sms_emoa(population, 2, ref_point=ref)
    assert [ind[0] for ind in chosen] == [2, 1]


def test_three_objectives(multi_obj, make):
    creator.create_type("SMS3_FIT", Fitness, weights=(1.0, 1.0, 1.0))
    creator.create_type("SMS3_IND", list, fitness=creator.__dict__["SMS3_FIT"])
    try:
        ind_cls = creator.__dict__["SMS3_IND"]
        population = [
            make(ind_cls, [0], (3.0, 1.0, 2.0)),
            make(ind_cls, [1], (1.0, 3.0, 2.0)),
            make(ind_cls, [2], (2.0, 2.0, 3.0)),
            make(ind_cls, [3], (1.0, 1.0, 1.0)),
        ]
        chosen = tools.sel_sms_emoa(population, 2)
        assert [ind[0] for ind in chosen] == [1, 2]
    finally:
        del creator.__dict__["SMS3_FIT"]
        del creator.__dict__["SMS3_IND"]


def test_steady_state_single_removal(multi_obj, make):
    parents = _critical_population(multi_obj, make)[:4]
    offspring = make(multi_obj, [99], (4.5, 4.5))
    pool = parents + [offspring]
    chosen = tools.sel_sms_emoa(pool, 4)
    assert [ind[0] for ind in chosen] == [0, 1, 2, 3]
    assert offspring not in chosen


def test_sms_emoa_zdt1():
    tools.rng.seed(8915)
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        dimensions = 5
        bound_low, bound_up = 0.0, 1.0
        survivors = 16
        generations = 100

        toolbox = Toolbox()
        toolbox.register("attr_float", tools.rng.uniform, bound_low, bound_up)
        toolbox.register(
            "individual",
            tools.init_repeat,
            creator.__dict__[IND],
            toolbox.attr_float,
            dimensions,
        )
        toolbox.register("population", tools.init_repeat, list, toolbox.individual)

        toolbox.register(
            "mate", tools.cx_simulated_binary_bounded, low=bound_low, up=bound_up, eta=20.0
        )
        toolbox.register(
            "mutate",
            tools.mut_polynomial_bounded,
            low=bound_low,
            up=bound_up,
            eta=20.0,
            mut_prob=1.0 / dimensions,
        )

        toolbox.register("evaluate", tools.bm_zdt_1)
        toolbox.register("select", tools.sel_sms_emoa)

        pop = toolbox.population(size=survivors)
        fitness = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitness, strict=False):
            ind.fitness.values = fit

        pop = toolbox.select(pop, len(pop))
        for _gen in range(1, generations):
            offspring = tools.sel_tournament_dcd(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2], strict=False):
                if tools.rng.random() <= 0.9:
                    toolbox.mate(ind1, ind2)

                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.is_valid()]
            fitness = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitness, strict=False):
                ind.fitness.values = fit

            pop = toolbox.select(pop + offspring, survivors)

        hv = tools.hypervolume(pop, [11.0, 11.0])

        assert hv > HV_THRESHOLD
        for ind in pop:
            assert not any(numpy.asarray(ind) < bound_low)
            assert not any(numpy.asarray(ind) > bound_up)
    finally:
        del creator.__dict__[FIT]
        del creator.__dict__[IND]
