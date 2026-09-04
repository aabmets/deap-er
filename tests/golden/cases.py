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
"""Seeded cases behind the golden-value tests.

Each ``*_case`` function returns the results of one seed as a JSON-friendly
mapping. The test modules compare those results against the stored data;
``_generate.py`` writes that data. Both go through this module, so a case is
defined exactly once.

Functions ending in ``_exact`` return values that are reproducible bit for
bit: comparisons, integer arithmetic, and the correctly rounded ``sqrt``.
Functions ending in ``_approx`` route through ``exp``, ``log``, ``sin``,
``cos``, fractional ``pow``, or LAPACK, any of which may differ by an ulp
between platforms, so their values are compared with a tolerance and are
returned flattened.
"""

import json
import operator
import random
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy
from deap_er import base, creator, gp, tools
from deap_er.benchmarks.moving_peaks import MovingPeaks

SEEDS = tuple(range(25))
DATA_DIR = Path(__file__).parent

_TYPE_NAMES = (
    "GOLDEN_FIT_MIN",
    "GOLDEN_IND_MIN",
    "GOLDEN_FIT_2",
    "GOLDEN_IND_2",
    "GOLDEN_FIT_4",
    "GOLDEN_IND_4",
    "GOLDEN_FIT_MO",
    "GOLDEN_IND_MO",
    "GOLDEN_FIT_GP",
    "GOLDEN_IND_GP",
)


def load(name: str) -> dict[str, Any]:
    """Read one stored golden data file.

    Args:
        name: File stem under ``tests/golden``.

    Returns:
        The stored results, keyed by seed.
    """
    return json.loads((DATA_DIR / f"{name}.json").read_text())


@contextmanager
def golden_types() -> Iterator[SimpleNamespace]:
    """Create the creator types the cases need, and remove them after.

    Yields:
        A namespace of the created individual classes.
    """
    creator.create("GOLDEN_FIT_MIN", base.Fitness, weights=(-1.0,))
    creator.create("GOLDEN_IND_MIN", list, fitness=creator.__dict__["GOLDEN_FIT_MIN"])
    creator.create("GOLDEN_FIT_2", base.Fitness, weights=(1.0, 1.0))
    creator.create("GOLDEN_IND_2", list, fitness=creator.__dict__["GOLDEN_FIT_2"])
    creator.create("GOLDEN_FIT_4", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0))
    creator.create("GOLDEN_IND_4", list, fitness=creator.__dict__["GOLDEN_FIT_4"])
    creator.create("GOLDEN_FIT_MO", base.Fitness, weights=(-1.0, -1.0))
    creator.create("GOLDEN_IND_MO", numpy.ndarray, fitness=creator.__dict__["GOLDEN_FIT_MO"])
    creator.create("GOLDEN_FIT_GP", base.Fitness, weights=(-1.0,))
    creator.create("GOLDEN_IND_GP", gp.PrimitiveTree, fitness=creator.__dict__["GOLDEN_FIT_GP"])
    try:
        yield SimpleNamespace(
            min_ind=creator.__dict__["GOLDEN_IND_MIN"],
            two_obj=creator.__dict__["GOLDEN_IND_2"],
            four_obj=creator.__dict__["GOLDEN_IND_4"],
            mo_ind=creator.__dict__["GOLDEN_IND_MO"],
            gp_ind=creator.__dict__["GOLDEN_IND_GP"],
        )
    finally:
        for name in _TYPE_NAMES:
            del creator.__dict__[name]


def _population(cls: Any, seed: int, count: int, genes: int, objectives: int) -> list[Any]:
    """Build a seeded population with random genes and fitness values.

    Each individual carries its position as ``idx`` so that selection
    results can be recorded compactly as indices.
    """
    random.seed(seed)
    population = []
    for index in range(count):
        ind = cls([random.uniform(0.0, 1.0) for _ in range(genes)])
        ind.fitness.values = tuple(random.uniform(0.0, 10.0) for _ in range(objectives))
        ind.idx = index
        population.append(ind)
    return population


def _indices(population: list[Any]) -> list[int]:
    """Return the origin index of every individual in a selection."""
    return [int(ind.idx) for ind in population]


def _bits(population: list[Any]) -> list[str]:
    """Return a binary population as one string per individual."""
    return ["".join(str(int(gene)) for gene in ind) for ind in population]


def _flat(*values: Any) -> list[float]:
    """Flatten nested numeric values into one list of floats."""
    out: list[float] = []
    for value in values:
        if isinstance(value, int | float | numpy.floating | numpy.integer):
            out.append(float(value))
        elif isinstance(value, numpy.ndarray):
            out.extend(float(v) for v in value.ravel())
        else:
            for item in value:
                out.extend(_flat(item))
    return out


def _untyped_pset() -> gp.PrimitiveSet:
    """Build the untyped primitive set used by the crossover cases."""
    pset = gp.PrimitiveSet("MAIN", 2)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.mul, 2)
    pset.add_primitive(operator.sub, 2)
    pset.add_primitive(operator.neg, 1)
    return pset


def _typed_pset() -> gp.PrimitiveSetTyped:
    """Build a single-return-type primitive set.

    One shared return type keeps ``random.choice(list(common_types))``
    deterministic; with two or more types the set iterates in an
    address-dependent order that varies between processes.
    """
    pset = gp.PrimitiveSetTyped("MAIN", [float, float], float)
    pset.add_primitive(operator.add, [float, float], float)
    pset.add_primitive(operator.mul, [float, float], float)
    pset.add_primitive(operator.neg, [float], float)
    return pset


def selection_case_exact(types: SimpleNamespace, seed: int) -> dict[str, Any]:
    """SPEA-II and lexicase selection results for one seed."""
    out: dict[str, Any] = {}

    for sel_count in (2, 4, 7, 10, 12):
        population = _population(types.two_obj, seed, 10, 3, 2)
        tools.seed(seed + 300000)
        out[f"spea2/{sel_count}"] = _indices(tools.sel_spea_2(population, sel_count))

    population = _population(types.four_obj, seed, 8, 3, 4)
    for label, args in (
        ("lexicase", ()),
        ("epsilon_auto", (None,)),
        ("epsilon_zero", (0.0,)),
        ("epsilon_half", (0.5,)),
    ):
        tools.seed(seed + 200000)
        if label == "lexicase":
            chosen = tools.sel_lexicase(population, 6)
        else:
            chosen = tools.sel_epsilon_lexicase(population, 6, *args)
        out[label] = _indices(chosen)

    return out


def gp_crossover_case_exact(types: SimpleNamespace, seed: int) -> dict[str, Any]:
    """One-point and leaf-biased GP crossover results for one seed."""
    out: dict[str, Any] = {}

    for name, pset in (("untyped", _untyped_pset()), ("typed", _typed_pset())):
        tools.seed(seed)
        first = types.gp_ind(gp.gen_full(pset, 1, 4))
        second = types.gp_ind(gp.gen_grow(pset, 1, 4))
        tools.seed(seed + 100000)
        gp.cx_one_point(first, second)
        out[f"one_point/{name}"] = [str(first), str(second)]

        tools.seed(seed)
        third = types.gp_ind(gp.gen_full(pset, 1, 4))
        fourth = types.gp_ind(gp.gen_grow(pset, 1, 4))
        tools.seed(seed + 100000)
        gp.cx_one_point_leaf_biased(third, fourth, 0.1 + (seed % 9) / 10.0)
        out[f"leaf_biased/{name}"] = [str(third), str(fourth)]

    return out


def _ea_toolbox() -> base.Toolbox:
    """Build a toolbox whose operators are all discrete."""
    toolbox = base.Toolbox()
    toolbox.register("mate", tools.cx_two_point)
    toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.2)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("evaluate", lambda ind: (float(sum(ind)),))
    return toolbox


def ea_drivers_case_exact(types: SimpleNamespace, seed: int) -> dict[str, Any]:
    """Final population, logbook, and hall of fame of each EA driver."""
    out: dict[str, Any] = {}

    for name in ("ea_simple", "ea_mu_plus_lambda", "ea_mu_comma_lambda"):
        toolbox = _ea_toolbox()
        random.seed(seed + 500000)
        population = [types.min_ind([random.randint(0, 1) for _ in range(8)]) for _ in range(12)]
        for ind in population:
            ind.fitness.values = (float(sum(ind)),)

        hof = tools.HallOfFame(3)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("max", max)

        tools.seed(seed + 600000)
        if name == "ea_simple":
            final, logbook = tools.ea_simple(toolbox, population, 5, 0.5, 0.3, hof, stats)
        elif name == "ea_mu_plus_lambda":
            final, logbook = tools.ea_mu_plus_lambda(
                toolbox, population, 5, 14, 10, 0.5, 0.3, hof, stats
            )
        else:
            final, logbook = tools.ea_mu_comma_lambda(
                toolbox, population, 5, 14, 10, 0.5, 0.3, hof, stats
            )

        out[f"{name}/population"] = _bits(final)
        out[f"{name}/logbook"] = [dict(sorted(record.items())) for record in logbook]
        out[f"{name}/hof"] = _bits(list(hof))

    return out


def operators_case_exact(types: SimpleNamespace, seed: int) -> dict[str, Any]:
    """Integer mutation results for one seed."""
    tools.seed(seed + 9000)
    size = 3 + seed % 6
    ind = types.min_ind([tools.rng.randint(0, 5) for _ in range(size)])
    tools.mut_uniform_int(ind, 0, 9, 0.5)
    return {"uniform_int": [int(gene) for gene in ind]}


def operators_case_approx(types: SimpleNamespace, seed: int) -> dict[str, list[float]]:
    """Bounded crossover and mutation results for one seed."""
    out: dict[str, list[float]] = {}
    size = 3 + seed % 6

    random.seed(seed)
    first = types.min_ind([random.uniform(0, 1) for _ in range(size)])
    second = types.min_ind([random.uniform(0, 1) for _ in range(size)])
    tools.seed(seed + 5000)
    tools.cx_simulated_binary_bounded(first, second, 2.0 + seed % 5, 0.0, 1.0)
    out["sbx_bounded_scalar"] = _flat(list(first), list(second))

    tools.seed(seed + 6000)
    third = types.min_ind([tools.rng.uniform(0, 1) for _ in range(size)])
    fourth = types.min_ind([tools.rng.uniform(0, 1) for _ in range(size)])
    tools.cx_simulated_binary_bounded(third, fourth, 20.0, [0.0] * size, [1.0] * size)
    out["sbx_bounded_sequence"] = _flat(list(third), list(fourth))

    tools.seed(seed + 7000)
    poly = types.min_ind([tools.rng.uniform(0, 1) for _ in range(size)])
    tools.mut_polynomial_bounded(poly, 20.0, 0.0, 1.0, 0.6)
    out["polynomial_bounded"] = _flat(list(poly))

    tools.seed(seed + 8000)
    gauss = types.min_ind([tools.rng.uniform(0, 1) for _ in range(size)])
    tools.mut_gaussian(gauss, 0.0, 1.0, 0.5)
    out["gaussian_scalar"] = _flat(list(gauss))

    tools.seed(seed + 8500)
    gauss_seq = types.min_ind([tools.rng.uniform(0, 1) for _ in range(size)])
    tools.mut_gaussian(gauss_seq, [0.0] * size, [1.0] * size, 0.5)
    out["gaussian_sequence"] = _flat(list(gauss_seq))

    return out


def benchmarks_case_exact(types: SimpleNamespace, seed: int) -> dict[str, Any]:
    """ZDT1 and ZDT2 values, which use only sqrt and basic arithmetic."""
    out: dict[str, Any] = {}
    random.seed(seed)
    for dim in (2, 3, 5, 10):
        vector = types.min_ind(random.uniform(0.01, 1.0) for _ in range(dim))
        out[f"zdt1/{dim}"] = list(tools.bm_zdt_1(vector))
        out[f"zdt2/{dim}"] = list(tools.bm_zdt_2(vector))
    return out


def benchmarks_case_approx(types: SimpleNamespace, seed: int) -> dict[str, list[float]]:
    """ZDT3, ZDT4, and ZDT6 values, which use sin, cos, and exp."""
    out: dict[str, list[float]] = {}
    random.seed(seed)
    for dim in (2, 3, 5, 10):
        vector = types.min_ind(random.uniform(0.01, 1.0) for _ in range(dim))
        out[f"zdt3/{dim}"] = _flat(tools.bm_zdt_3(vector))
        out[f"zdt4/{dim}"] = _flat(tools.bm_zdt_4(vector))
        out[f"zdt6/{dim}"] = _flat(tools.bm_zdt_6(vector))
    return out


def moving_peaks_case_approx(_types: SimpleNamespace, seed: int) -> dict[str, list[float]]:
    """Moving-peaks landscape state after three changes."""
    out: dict[str, list[float]] = {}

    dim = 2 + seed % 3
    tools.seed(seed)
    fluctuating = MovingPeaks(dimensions=dim, npeaks=[2, 4, 7], change_severity=1.0)
    for _ in range(3):
        fluctuating.change_peaks()
    out["fluctuating"] = _flat(
        fluctuating.peaks_position, fluctuating.peaks_height, fluctuating.peaks_width
    )
    out["fluctuating_eval"] = _flat(fluctuating([0.5] * dim))

    tools.seed(seed + 400000)
    fixed = MovingPeaks(dimensions=3)
    for _ in range(3):
        fixed.change_peaks()
    out["fixed"] = _flat(fixed.peaks_position, fixed.peaks_height, fixed.peaks_width)

    return out


def _clipped_zdt1(individual: Any) -> tuple[float, float]:
    """Evaluate ZDT1 on an individual clipped into the unit box."""
    return tools.bm_zdt_1(numpy.clip(numpy.asarray(individual), 0.0, 1.0))


def strategies_case_approx(types: SimpleNamespace, seed: int) -> dict[str, list[float]]:
    """State of each CMA strategy after a short run."""
    out: dict[str, list[float]] = {}

    tools.seed(seed)
    standard = tools.Strategy(centroid=[0.0] * 5, sigma=1.0)
    for _ in range(10):
        population = standard.generate(types.min_ind)
        for ind in population:
            ind.fitness.values = tools.bm_sphere(ind)
        standard.update(population)
    out["standard"] = _flat(
        standard.centroid, standard.sigma, standard.diag_d, standard.pc, standard.ps
    )

    tools.seed(seed + 900)
    parent = types.min_ind([1.0] * 4)
    parent.fitness.values = tools.bm_sphere(parent)
    one_plus = tools.StrategyOnePlusLambda(parent, sigma=0.5, offsprings=3)
    for _ in range(12):
        population = one_plus.generate(types.min_ind)
        for ind in population:
            ind.fitness.values = tools.bm_sphere(ind)
        one_plus.update(population)
    out["one_plus_lambda"] = _flat(
        list(one_plus.parent), one_plus.sigma, one_plus.psucc, one_plus.big_a, one_plus.pc
    )

    tools.seed(seed + 1900)
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
    out["multi_objective"] = _flat(
        multi.sigmas, multi.psucc, [numpy.asarray(p) for p in multi.parents]
    )
    out["multi_objective_hv"] = _flat(tools.hypervolume(multi.parents, [11.0, 11.0]))

    tools.seed(seed + 2900)
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
    out["multi_objective_uneven"] = _flat(uneven.sigmas, uneven.psucc)

    return out


CASES = {
    "selection_exact": selection_case_exact,
    "gp_crossover_exact": gp_crossover_case_exact,
    "ea_drivers_exact": ea_drivers_case_exact,
    "operators_exact": operators_case_exact,
    "operators_approx": operators_case_approx,
    "benchmarks_exact": benchmarks_case_exact,
    "benchmarks_approx": benchmarks_case_approx,
    "moving_peaks_approx": moving_peaks_case_approx,
    "strategies_approx": strategies_case_approx,
}
