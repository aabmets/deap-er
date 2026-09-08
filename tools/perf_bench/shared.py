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
"""Hot-path cases shared with DEAP/deap."""

from __future__ import annotations

import statistics
import time
from copy import deepcopy

from .adapters import DeapErLib, DeapLib, Library
from .data import gp_exprs, mo_data, optimal_front, so_data
from .report import CaseResult
from .timing import LIB_DEAP, LIB_ER, REPEAT, mean_ms

_DESCRIPTIONS = {
    "fitness.values x200 on n=80": "Cached fitness.values reads",
    "fitness.dominates pairwise n=80": "Pairwise Fitness.dominates",
    "ParetoFront.update n=80": "ParetoFront.update on a pool",
    "sel_spea_2 n=80 k=40": "SPEA-II selection",
    "sel_spea_2 n=160 k=80": "SPEA-II selection (larger)",
    "nsga_convergence n=40": "NSGA convergence metric",
    "sel_nsga_2 n=80 k=40": "NSGA-II selection",
    "sel_nsga_3 n=80 k=40": "NSGA-III selection",
    "compile_tree 40 unique trees": "First compile of 40 GP trees",
    "compile_tree 10 trees x20 repeats": "Cached compile of 10 GP trees",
    "sel_tournament n=80 k=80": "Tournament selection",
    "sel_lexicase n=200 k=100 cases=500": "Lexicase selection",
    "deepcopy n=60 list inds": "copy.deepcopy of list individuals",
    "clone_individual n=60 list inds": "clone_individual vs deepcopy",
    "ea_simple n=40 gens=8": "ea_simple OneMax loop",
}


def _measure_compile(lib: Library, trees: list, pset: object) -> tuple[float, float]:
    """Time a cold compile of 40 trees and a cached 10-tree × 20 repeat.

    Args:
        lib: Library adapter.
        trees: Shared expression strings.
        pset: Primitive set used to compile them.

    Returns:
        ``(first_compile_ms, cached_repeat_ms)``.
    """

    def compile_once() -> None:
        lib.clear_compile_cache()
        for tree in trees:
            lib.compile_one(tree, pset)(0.5)

    def compile_repeat_ms() -> float:
        lib.clear_compile_cache()
        for tree in trees[:10]:
            lib.compile_one(tree, pset)(0.5)
        start = time.perf_counter()
        for _ in range(20):
            for tree in trees[:10]:
                lib.compile_one(tree, pset)(0.5)
        return (time.perf_counter() - start) * 1000.0

    first_ms = mean_ms(compile_once, warmup=0)
    cached_ms = statistics.fmean([compile_repeat_ms() for _ in range(REPEAT)])
    return first_ms, cached_ms


def run_library(lib: Library) -> dict[str, float]:
    """Run every shared hot-path case against one library.

    Args:
        lib: Library adapter.

    Returns:
        Case name to mean milliseconds.
    """
    lib.make_types()
    results: dict[str, float] = {}
    genes80, fits80 = mo_data(80, 10, 3, 1)
    genes160, fits160 = mo_data(160, 10, 3, 2)
    genes_front, fits_front = mo_data(40, 8, 3, 3)
    pop80 = lib.wrap_mo(genes80, fits80)
    pop160 = lib.wrap_mo(genes160, fits160)
    front = lib.wrap_mo(genes_front, fits_front)
    optimal = optimal_front(40, 3, 4)

    def values_reads() -> None:
        total = 0.0
        for _ in range(200):
            for individual in pop80:
                values = individual.fitness.values
                total += values[0] + values[1] + values[2]
        if total == 0.0:
            raise RuntimeError("unreachable")

    def pairwise_dominates() -> None:
        count = 0
        for index, left in enumerate(pop80):
            for right in pop80[index + 1 :]:
                count += left.fitness.dominates(right.fitness)
                count += right.fitness.dominates(left.fitness)
        if count < 0:
            raise RuntimeError("unreachable")

    refs = lib.reference_points()
    results["fitness.values x200 on n=80"] = mean_ms(values_reads)
    results["fitness.dominates pairwise n=80"] = mean_ms(pairwise_dominates)
    results["ParetoFront.update n=80"] = mean_ms(lambda: lib.new_pareto().update(pop80))
    results["sel_spea_2 n=80 k=40"] = mean_ms(lambda: (lib.seed(11), lib.sel_spea(pop80, 40)))
    results["sel_spea_2 n=160 k=80"] = mean_ms(lambda: (lib.seed(12), lib.sel_spea(pop160, 80)))
    results["nsga_convergence n=40"] = mean_ms(lambda: lib.convergence(front, optimal))
    results["sel_nsga_2 n=80 k=40"] = mean_ms(lambda: lib.sel_nsga2(pop80, 40))
    results["sel_nsga_3 n=80 k=40"] = mean_ms(lambda: lib.sel_nsga3(pop80, 40, refs))

    pset = lib.make_pset()
    first_ms, cached_ms = _measure_compile(lib, gp_exprs(40, 21), pset)
    results["compile_tree 40 unique trees"] = first_ms
    results["compile_tree 10 trees x20 repeats"] = cached_ms

    so_pop = lib.wrap_so(so_data(60, 32, 31))
    so80 = lib.wrap_so(so_data(80, 32, 32))
    results["sel_tournament n=80 k=80"] = mean_ms(
        lambda: (lib.seed(33), lib.sel_tournament(so80, 80, 3))
    )
    pop_lc = lib.wrap_many_case(*mo_data(200, 10, 500, 34))
    results["sel_lexicase n=200 k=100 cases=500"] = mean_ms(
        lambda: (lib.seed(35), lib.sel_lexicase(pop_lc, 100))
    )
    results["deepcopy n=60 list inds"] = mean_ms(
        lambda: [deepcopy(individual) for individual in so_pop]
    )
    results["clone_individual n=60 list inds"] = mean_ms(
        lambda: [lib.clone_one(individual) for individual in so_pop]
    )
    results["ea_simple n=40 gens=8"] = mean_ms(
        lambda: (lib.seed(41), lib.run_ea(lib.wrap_so(so_data(40, 24, 41))))
    )
    lib.drop_types()
    return results


def run_shared_cases() -> list[CaseResult]:
    """Time every shared case on DEAP and deap-er.

    Returns:
        One ``CaseResult`` per shared component.
    """
    timings = {LIB_DEAP: run_library(DeapLib()), LIB_ER: run_library(DeapErLib())}
    cases = []
    for name in timings[LIB_DEAP]:
        warmup = 0 if name.startswith("compile_tree") else 2
        feature = "clone_individual" if "clone_individual" in name else ""
        cases.append(
            CaseResult(
                name=name,
                description=_DESCRIPTIONS.get(name, name),
                shared=True,
                deap_er_ms=timings[LIB_ER][name],
                deap_ms=timings[LIB_DEAP][name],
                feature=feature,
                warmup=warmup,
            )
        )
    return cases
