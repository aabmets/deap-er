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
import random
from types import SimpleNamespace
from typing import Any

import numpy
from deap_er import tools
from deap_er.benchmarks import MovingPeaks
from tests.golden.case_support import flatten_values


def operators_case_approx(types: SimpleNamespace, seed: int) -> dict[str, list[float]]:
    """Bounded crossover and mutation results for one seed."""
    out: dict[str, list[float]] = {}
    size = 3 + seed % 6

    random.seed(seed)
    first = types.min_ind([random.uniform(0, 1) for _ in range(size)])
    second = types.min_ind([random.uniform(0, 1) for _ in range(size)])
    tools.rng.seed(seed + 5000)
    tools.cx_simulated_binary_bounded(first, second, 2.0 + seed % 5, 0.0, 1.0)
    out["sbx_bounded_scalar"] = flatten_values(list(first), list(second))

    tools.rng.seed(seed + 6000)
    third = types.min_ind([tools.rng.uniform(0, 1) for _ in range(size)])
    fourth = types.min_ind([tools.rng.uniform(0, 1) for _ in range(size)])
    tools.cx_simulated_binary_bounded(third, fourth, 20.0, [0.0] * size, [1.0] * size)
    out["sbx_bounded_sequence"] = flatten_values(list(third), list(fourth))

    tools.rng.seed(seed + 7000)
    poly = types.min_ind([tools.rng.uniform(0, 1) for _ in range(size)])
    tools.mut_polynomial_bounded(poly, 20.0, 0.0, 1.0, 0.6)
    out["polynomial_bounded"] = flatten_values(list(poly))

    tools.rng.seed(seed + 8000)
    gauss = types.min_ind([tools.rng.uniform(0, 1) for _ in range(size)])
    tools.mut_gaussian(gauss, 0.0, 1.0, 0.5)
    out["gaussian_scalar"] = flatten_values(list(gauss))

    tools.rng.seed(seed + 8500)
    gauss_seq = types.min_ind([tools.rng.uniform(0, 1) for _ in range(size)])
    tools.mut_gaussian(gauss_seq, [0.0] * size, [1.0] * size, 0.5)
    out["gaussian_sequence"] = flatten_values(list(gauss_seq))

    return out


def benchmarks_case_approx(types: SimpleNamespace, seed: int) -> dict[str, list[float]]:
    """ZDT3, ZDT4, and ZDT6 values, which use sin, cos, and exp."""
    out: dict[str, list[float]] = {}
    random.seed(seed)
    for dim in (2, 3, 5, 10):
        vector = types.min_ind(random.uniform(0.01, 1.0) for _ in range(dim))
        out[f"zdt3/{dim}"] = flatten_values(tools.bm_zdt_3(vector))
        out[f"zdt4/{dim}"] = flatten_values(tools.bm_zdt_4(vector))
        out[f"zdt6/{dim}"] = flatten_values(tools.bm_zdt_6(vector))
    return out


def moving_peaks_case_approx(_types: SimpleNamespace, seed: int) -> dict[str, list[float]]:
    """Moving-peaks landscape state after three changes."""
    out: dict[str, list[float]] = {}

    dim = 2 + seed % 3
    tools.rng.seed(seed)
    fluctuating = MovingPeaks(dimensions=dim, npeaks=[2, 4, 7], change_severity=1.0)
    for _ in range(3):
        fluctuating.change_peaks()
    out["fluctuating"] = flatten_values(
        fluctuating.peaks_position, fluctuating.peaks_height, fluctuating.peaks_width
    )
    out["fluctuating_eval"] = flatten_values(fluctuating([0.5] * dim))

    tools.rng.seed(seed + 400000)
    fixed = MovingPeaks(dimensions=3)
    for _ in range(3):
        fixed.change_peaks()
    out["fixed"] = flatten_values(fixed.peaks_position, fixed.peaks_height, fixed.peaks_width)

    return out


def _clipped_zdt1(individual: Any) -> tuple[float, float]:
    """Evaluate ZDT1 on an individual clipped into the unit box."""
    return tools.bm_zdt_1(numpy.clip(numpy.asarray(individual), 0.0, 1.0))


def strategies_case_approx(types: SimpleNamespace, seed: int) -> dict[str, list[float]]:
    """State of each CMA strategy after a short run."""
    out: dict[str, list[float]] = {}

    tools.rng.seed(seed)
    standard = tools.Strategy(centroid=[0.0] * 5, sigma=1.0)
    for _ in range(10):
        population = standard.generate(types.min_ind)
        for ind in population:
            ind.fitness.values = tools.bm_sphere(ind)
        standard.update(population)
    out["standard"] = flatten_values(
        standard.centroid, standard.sigma, standard.diag_d, standard.pc, standard.ps
    )

    tools.rng.seed(seed + 900)
    parent = types.min_ind([1.0] * 4)
    parent.fitness.values = tools.bm_sphere(parent)
    one_plus = tools.StrategyOnePlusLambda(parent, sigma=0.5, offsprings=3)
    for _ in range(12):
        population = one_plus.generate(types.min_ind)
        for ind in population:
            ind.fitness.values = tools.bm_sphere(ind)
        one_plus.update(population)
    out["one_plus_lambda"] = flatten_values(
        list(one_plus.parent), one_plus.sigma, one_plus.psucc, one_plus.big_a, one_plus.pc
    )

    tools.rng.seed(seed + 1900)
    choices = [[tools.rng.uniform(0.0, 1.0) for _ in range(4)] for _ in range(6)]
    mo_population = [types.mo_ind(x) for x in choices]
    for ind in mo_population:
        ind.fitness.values = _clipped_zdt1(ind)
    multi = tools.StrategyMultiObjective(mo_population, sigma=1.0, survivors=6, offsprings=6)
    for _ in range(20):
        population = multi.generate(types.mo_ind)
        for ind in population:
            ind.fitness.values = _clipped_zdt1(ind)
        multi.update(population)
    out["multi_objective"] = flatten_values(
        multi.sigmas, multi.psucc, [numpy.asarray(p) for p in multi.parents]
    )
    out["multi_objective_hv"] = flatten_values(tools.hypervolume(multi.parents, [11.0, 11.0]))

    tools.rng.seed(seed + 2900)
    choices = [[tools.rng.uniform(0.0, 1.0) for _ in range(3)] for _ in range(5)]
    uneven_population = [types.mo_ind(x) for x in choices]
    for ind in uneven_population:
        ind.fitness.values = _clipped_zdt1(ind)
    uneven = tools.StrategyMultiObjective(uneven_population, sigma=0.8, survivors=3, offsprings=7)
    for _ in range(12):
        population = uneven.generate(types.mo_ind)
        for ind in population:
            ind.fitness.values = _clipped_zdt1(ind)
        uneven.update(population)
    out["multi_objective_uneven"] = flatten_values(uneven.sigmas, uneven.psucc)

    return out
