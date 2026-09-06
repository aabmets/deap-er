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
"""Compare hot-path timings of DEAP/deap and aabmets/deap-er.

Arithmetic mean of 30 timed runs after 2 warmups, except first-time
compile (no warmup; deap-er cache cleared each sample). Populations are
built from the same numeric genomes so both libraries do the same work.

    uv run python tools/bench_hotpaths.py
"""

import argparse
import json
import random
import statistics
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

import deap
from deap import algorithms as deap_algorithms
from deap import base as deap_base
from deap import creator as deap_creator
from deap import gp as deap_gp
from deap import tools as deap_tools
from deap.benchmarks.tools import convergence as deap_convergence
from deap_er import Fitness as ErFitness
from deap_er import Toolbox as ErToolbox
from deap_er import creator as er_creator
from deap_er import gp as er_gp
from deap_er import tools as er_tools

REPEAT = 30
WARMUP = 2
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_JSON = _REPO_ROOT / "reports" / "hotpath-bench.json"
_TYPE_NAMES = ("B_FIT", "B_IND", "B_FIT_SO", "B_IND_SO", "B_FIT_GP")
_LIB_DEAP = "DEAP/deap"
_LIB_ER = "aabmets/deap-er"


class _Library(Protocol):
    """Minimal surface the shared benches need from one EA library."""

    key: str
    label: str

    def seed(self, value: int) -> None: ...
    def make_types(self) -> None: ...
    def drop_types(self) -> None: ...
    def wrap_mo(self, genomes: list[list[float]], fits: list[tuple]) -> list: ...
    def wrap_so(self, genomes: list[list[int]]) -> list: ...
    def sel_spea(self, population: list, count: int) -> object: ...
    def sel_nsga3(self, population: list, count: int, refs: object) -> object: ...
    def reference_points(self) -> object: ...
    def convergence(self, front: list, optimal: list[tuple]) -> object: ...
    def make_pset(self) -> object: ...
    def make_trees(self, pset: object, count: int) -> list: ...
    def compile_one(self, tree: object, pset: object) -> Callable[[float], object]: ...
    def clear_compile_cache(self) -> None: ...
    def clone_one(self, individual: object) -> object: ...
    def run_ea(self, population: list) -> object: ...


def _mean_ms(
    func: Callable[[], object],
    *,
    repeat: int = REPEAT,
    warmup: int = WARMUP,
) -> float:
    """Return the arithmetic-mean runtime of ``func`` in milliseconds.

    Args:
        func: Nullary callable to time.
        repeat: Timed repetitions after warmup.
        warmup: Untimed calls to run first.

    Returns:
        Mean elapsed milliseconds.
    """
    for _ in range(warmup):
        func()
    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.fmean(samples)


def _drop_creator_types(module: Any) -> None:
    """Remove bench types from a creator module.

    Args:
        module: ``creator`` module that may hold the bench types.
    """
    for name in _TYPE_NAMES:
        if name in module.__dict__:
            delattr(module, name)


class _DeapLib:
    """Adapter for upstream DEAP."""

    key = "deap"
    label = _LIB_DEAP

    def seed(self, value: int) -> None:
        random.seed(value)

    def make_types(self) -> None:
        deap_creator.create("B_FIT", deap_base.Fitness, weights=(1.0, -1.0, 1.0))
        deap_creator.create("B_IND", list, fitness=deap_creator.B_FIT)
        deap_creator.create("B_FIT_SO", deap_base.Fitness, weights=(-1.0,))
        deap_creator.create("B_IND_SO", list, fitness=deap_creator.B_FIT_SO)
        deap_creator.create("B_FIT_GP", deap_base.Fitness, weights=(-1.0,))

    def drop_types(self) -> None:
        _drop_creator_types(deap_creator)

    def wrap_mo(self, genomes: list[list[float]], fits: list[tuple]) -> list:
        population = []
        for genome, fit in zip(genomes, fits, strict=True):
            individual = deap_creator.B_IND(genome)
            individual.fitness.values = fit
            population.append(individual)
        return population

    def wrap_so(self, genomes: list[list[int]]) -> list:
        population = []
        for genome in genomes:
            individual = deap_creator.B_IND_SO(genome)
            individual.fitness.values = (float(sum(genome)),)
            population.append(individual)
        return population

    def sel_spea(self, population: list, count: int) -> object:
        return deap_tools.selSPEA2(population, count)

    def sel_nsga3(self, population: list, count: int, refs: object) -> object:
        return deap_tools.selNSGA3(population, count, refs)

    def reference_points(self) -> object:
        return deap_tools.uniform_reference_points(3, p=6)

    def convergence(self, front: list, optimal: list[tuple]) -> object:
        return deap_convergence(front, optimal)

    def make_pset(self) -> object:
        pset = deap_gp.PrimitiveSet("MAIN", 1)
        pset.addPrimitive(lambda a, b: a + b, 2, name="add")
        pset.addPrimitive(lambda a, b: a * b, 2, name="mul")
        pset.addPrimitive(lambda a: -a, 1, name="neg")
        return pset

    def make_trees(self, pset: object, count: int) -> list:
        self.seed(21)
        return [deap_gp.PrimitiveTree(deap_gp.genHalfAndHalf(pset, 1, 4)) for _ in range(count)]

    def compile_one(self, tree: object, pset: object) -> Callable[[float], object]:
        return deap_gp.compile(tree, pset)

    def clear_compile_cache(self) -> None:
        return

    def clone_one(self, individual: object) -> object:
        return deepcopy(individual)

    def run_ea(self, population: list) -> object:
        toolbox = deap_base.Toolbox()
        toolbox.register("mate", deap_tools.cxTwoPoint)
        toolbox.register("mutate", deap_tools.mutFlipBit, indpb=0.2)
        toolbox.register("select", deap_tools.selTournament, tournsize=3)
        toolbox.register("evaluate", lambda individual: (float(sum(individual)),))
        toolbox.register("clone", deepcopy)
        return deap_algorithms.eaSimple(
            population, toolbox, cxpb=0.5, mutpb=0.2, ngen=8, verbose=False
        )


class _DeapErLib:
    """Adapter for deap-er."""

    key = "deap-er"
    label = _LIB_ER

    def seed(self, value: int) -> None:
        er_tools.rng.seed(value)

    def make_types(self) -> None:
        er_creator.create_type("B_FIT", ErFitness, weights=(1.0, -1.0, 1.0))
        er_creator.create_type("B_IND", list, fitness=er_creator.B_FIT)
        er_creator.create_type("B_FIT_SO", ErFitness, weights=(-1.0,))
        er_creator.create_type("B_IND_SO", list, fitness=er_creator.B_FIT_SO)
        er_creator.create_type("B_FIT_GP", ErFitness, weights=(-1.0,))

    def drop_types(self) -> None:
        _drop_creator_types(er_creator)

    def wrap_mo(self, genomes: list[list[float]], fits: list[tuple]) -> list:
        population = []
        for genome, fit in zip(genomes, fits, strict=True):
            individual = er_creator.B_IND(genome)
            individual.fitness.values = fit
            population.append(individual)
        return population

    def wrap_so(self, genomes: list[list[int]]) -> list:
        population = []
        for genome in genomes:
            individual = er_creator.B_IND_SO(genome)
            individual.fitness.values = (float(sum(genome)),)
            population.append(individual)
        return population

    def sel_spea(self, population: list, count: int) -> object:
        return er_tools.sel_spea_2(population, count)

    def sel_nsga3(self, population: list, count: int, refs: object) -> object:
        return er_tools.sel_nsga_3(population, count, refs)

    def reference_points(self) -> object:
        return er_tools.uniform_reference_points(3, ref_ppo=6)

    def convergence(self, front: list, optimal: list[tuple]) -> object:
        return er_tools.nsga_convergence(front, optimal)

    def make_pset(self) -> object:
        pset = er_gp.PrimitiveSet("MAIN", 1)
        pset.add_primitive(lambda a, b: a + b, 2, name="add")
        pset.add_primitive(lambda a, b: a * b, 2, name="mul")
        pset.add_primitive(lambda a: -a, 1, name="neg")
        return pset

    def make_trees(self, pset: object, count: int) -> list:
        self.seed(21)
        return [er_gp.PrimitiveTree(er_gp.gen_half_and_half(pset, 1, 4)) for _ in range(count)]

    def compile_one(self, tree: object, pset: object) -> Callable[[float], object]:
        return er_gp.compile_tree(tree, pset)

    def clear_compile_cache(self) -> None:
        module = sys.modules.get("deap_er.private.programming.compilers")
        cache = getattr(module, "_compile_cache", None)
        if isinstance(cache, dict):
            cache.clear()

    def clone_one(self, individual: object) -> object:
        return er_tools.clone_individual(individual)

    def run_ea(self, population: list) -> object:
        toolbox = ErToolbox()
        toolbox.register("mate", er_tools.cx_two_point)
        toolbox.register("mutate", er_tools.mut_flip_bit, mut_prob=0.2)
        toolbox.register("select", er_tools.sel_tournament, contestants=3)
        toolbox.register("evaluate", lambda individual: (float(sum(individual)),))
        return er_tools.ea_simple(toolbox, population, 8, 0.5, 0.2)


def _mo_data(
    count: int, genes: int, objectives: int, seed: int
) -> tuple[list[list[float]], list[tuple]]:
    """Build shared multi-objective genomes and fitness tuples.

    Args:
        count: Number of individuals.
        genes: Gene count per individual.
        objectives: Fitness objective count.
        seed: RNG seed.

    Returns:
        Genomes and matching fitness tuples.
    """
    rng = random.Random(seed)
    genomes = [[rng.random() for _ in range(genes)] for _ in range(count)]
    fits = [tuple(rng.random() for _ in range(objectives)) for _ in range(count)]
    return genomes, fits


def _so_data(count: int, genes: int, seed: int) -> list[list[int]]:
    """Build shared OneMax-style binary genomes.

    Args:
        count: Number of individuals.
        genes: Bit count per individual.
        seed: RNG seed.

    Returns:
        Binary genomes.
    """
    rng = random.Random(seed)
    return [[rng.randint(0, 1) for _ in range(genes)] for _ in range(count)]


def _optimal_front(count: int, objectives: int, seed: int) -> list[tuple]:
    """Build shared optimal-front coordinates.

    Args:
        count: Number of reference points.
        objectives: Coordinate count.
        seed: RNG seed.

    Returns:
        Optimal-front tuples.
    """
    rng = random.Random(seed)
    return [tuple(rng.random() for _ in range(objectives)) for _ in range(count)]


def _measure_compile(lib: _Library, trees: list, pset: object) -> tuple[float, float]:
    """Time a cold compile of 40 trees and a cached 10-tree × 20 repeat.

    Args:
        lib: Library adapter.
        trees: Pre-generated primitive trees.
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

    first_ms = _mean_ms(compile_once, warmup=0)
    cached_ms = statistics.fmean([compile_repeat_ms() for _ in range(REPEAT)])
    return first_ms, cached_ms


def _run_library(lib: _Library) -> dict[str, float]:
    """Run every hot-path case against one library.

    Args:
        lib: Library adapter.

    Returns:
        Case name to mean milliseconds.
    """
    lib.make_types()
    results: dict[str, float] = {}

    genes80, fits80 = _mo_data(80, 10, 3, 1)
    genes160, fits160 = _mo_data(160, 10, 3, 2)
    genes_front, fits_front = _mo_data(40, 8, 3, 3)
    pop80 = lib.wrap_mo(genes80, fits80)
    pop160 = lib.wrap_mo(genes160, fits160)
    front = lib.wrap_mo(genes_front, fits_front)
    optimal = _optimal_front(40, 3, 4)

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

    def spea80() -> None:
        lib.seed(11)
        lib.sel_spea(pop80, 40)

    def spea160() -> None:
        lib.seed(12)
        lib.sel_spea(pop160, 80)

    def conv() -> None:
        lib.convergence(front, optimal)

    refs = lib.reference_points()

    def nsga3() -> None:
        lib.sel_nsga3(pop80, 40, refs)

    results["fitness.values x200 on n=80"] = _mean_ms(values_reads)
    results["fitness.dominates pairwise n=80"] = _mean_ms(pairwise_dominates)
    results["sel_spea_2 n=80 k=40"] = _mean_ms(spea80)
    results["sel_spea_2 n=160 k=80"] = _mean_ms(spea160)
    results["nsga_convergence n=40"] = _mean_ms(conv)
    results["sel_nsga_3 n=80 k=40"] = _mean_ms(nsga3)

    pset = lib.make_pset()
    trees = lib.make_trees(pset, 40)
    first_ms, cached_ms = _measure_compile(lib, trees, pset)
    results["compile_tree 40 unique trees"] = first_ms
    results["compile_tree 10 trees x20 repeats"] = cached_ms

    so_pop = lib.wrap_so(_so_data(60, 32, 31))

    def deepcopies() -> None:
        for individual in so_pop:
            deepcopy(individual)

    def clones() -> None:
        for individual in so_pop:
            lib.clone_one(individual)

    results["deepcopy n=60 list inds"] = _mean_ms(deepcopies)
    results["clone_individual n=60 list inds"] = _mean_ms(clones)

    def ea_run() -> None:
        lib.seed(41)
        lib.run_ea(lib.wrap_so(_so_data(40, 24, 41)))

    results["ea_simple n=40 gens=8"] = _mean_ms(ea_run)
    lib.drop_types()
    return results


def _package_version(name: str) -> str:
    """Return an installed package version, or ``unknown``.

    Args:
        name: Distribution name on PyPI.

    Returns:
        Version string.
    """
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _build_report(
    timings: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Assemble the JSON document from per-library timings.

    Args:
        timings: Map of library label to case-name timings.

    Returns:
        JSON-serializable report.
    """
    deap_ms = timings[_LIB_DEAP]
    er_ms = timings[_LIB_ER]
    cases = []
    relative: dict[str, float] = {}
    for name in deap_ms:
        speed = deap_ms[name] / er_ms[name]
        relative[name] = speed
        cases.append(
            {
                "name": name,
                "deap_ms": deap_ms[name],
                "deap_er_ms": er_ms[name],
                "relative_speed": speed,
            }
        )
    return {
        "meta": {
            "deap_version": _package_version("deap"),
            "deap_er_version": _package_version("deap-er"),
            "repeat": REPEAT,
            "warmup": WARMUP,
            "deap_module": deap.__file__,
        },
        "libraries": timings,
        "relative_speed": relative,
        "cases": cases,
    }


def _print_table(report: dict[str, object]) -> None:
    """Print a comparison table to stdout.

    Args:
        report: Document from ``_build_report``.
    """
    print(f"{'Case':<38} {_LIB_DEAP:>12} {_LIB_ER:>20} {'Speed':>8}")
    for case in report["cases"]:
        print(
            f"{case['name']:<38} {case['deap_ms']:>11.4f} "
            f"{case['deap_er_ms']:>19.4f} {case['relative_speed']:>7.2f}×"
        )


def _write_json(report: dict[str, object], path: Path) -> None:
    """Write the report as indented JSON.

    Args:
        report: Document from ``_build_report``.
        path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Time both libraries, print a table, and write JSON.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_JSON,
        help=f"JSON path (default: {_DEFAULT_JSON})",
    )
    args = parser.parse_args(argv)

    timings = {
        _LIB_DEAP: _run_library(_DeapLib()),
        _LIB_ER: _run_library(_DeapErLib()),
    }
    report = _build_report(timings)
    _print_table(report)
    _write_json(report, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
