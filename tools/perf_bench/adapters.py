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
# ruff: noqa: D102
"""Library adapters so shared benches call DEAP and deap-er the same way."""

from __future__ import annotations

import random
import sys
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Protocol

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

from .data import drop_creator_types
from .timing import LIB_DEAP, LIB_ER


class Library(Protocol):
    """Minimal surface the shared benches need from one EA library."""

    key: str
    label: str

    def seed(self, value: int) -> None: ...
    def make_types(self) -> None: ...
    def drop_types(self) -> None: ...
    def wrap_mo(self, genomes: list[list[float]], fits: list[tuple]) -> list: ...
    def wrap_many_case(self, genomes: list[list[float]], fits: list[tuple]) -> list: ...
    def wrap_so(self, genomes: list[list[int]]) -> list: ...
    def sel_spea(self, population: list, count: int) -> object: ...
    def sel_nsga2(self, population: list, count: int) -> object: ...
    def sel_nsga3(self, population: list, count: int, refs: object) -> object: ...
    def sel_tournament(self, population: list, count: int, contestants: int) -> object: ...
    def sel_lexicase(self, population: list, count: int) -> object: ...
    def new_pareto(self) -> Any: ...
    def reference_points(self) -> object: ...
    def convergence(self, front: list, optimal: list[tuple]) -> object: ...
    def make_pset(self) -> object: ...
    def compile_one(self, tree: object, pset: object) -> Callable[[float], object]: ...
    def clear_compile_cache(self) -> None: ...
    def clone_one(self, individual: object) -> object: ...
    def run_ea(self, population: list) -> object: ...


class DeapLib:
    """Adapter for upstream DEAP."""

    key = "deap"
    label = LIB_DEAP

    def seed(self, value: int) -> None:
        random.seed(value)

    def make_types(self) -> None:
        deap_creator.create("B_FIT", deap_base.Fitness, weights=(1.0, -1.0, 1.0))
        deap_creator.create("B_IND", list, fitness=deap_creator.B_FIT)
        deap_creator.create("B_FIT_SO", deap_base.Fitness, weights=(-1.0,))
        deap_creator.create("B_IND_SO", list, fitness=deap_creator.B_FIT_SO)
        deap_creator.create("B_FIT_GP", deap_base.Fitness, weights=(-1.0,))
        deap_creator.create("B_FIT_LC", deap_base.Fitness, weights=tuple(-1.0 for _ in range(500)))
        deap_creator.create("B_IND_LC", list, fitness=deap_creator.B_FIT_LC)

    def drop_types(self) -> None:
        drop_creator_types(deap_creator)

    def wrap_mo(self, genomes: list[list[float]], fits: list[tuple]) -> list:
        return _wrap(deap_creator.B_IND, genomes, fits)

    def wrap_many_case(self, genomes: list[list[float]], fits: list[tuple]) -> list:
        return _wrap(deap_creator.B_IND_LC, genomes, fits)

    def wrap_so(self, genomes: list[list[int]]) -> list:
        return _wrap_so(deap_creator.B_IND_SO, genomes)

    def sel_spea(self, population: list, count: int) -> object:
        return deap_tools.selSPEA2(population, count)

    def sel_nsga2(self, population: list, count: int) -> object:
        return deap_tools.selNSGA2(population, count)

    def sel_nsga3(self, population: list, count: int, refs: object) -> object:
        return deap_tools.selNSGA3(population, count, refs)

    def sel_tournament(self, population: list, count: int, contestants: int) -> object:
        return deap_tools.selTournament(population, count, tournsize=contestants)

    def sel_lexicase(self, population: list, count: int) -> object:
        return deap_tools.selLexicase(population, count)

    def new_pareto(self) -> Any:
        return deap_tools.ParetoFront()

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


class DeapErLib:
    """Adapter for deap-er."""

    key = "deap-er"
    label = LIB_ER

    def seed(self, value: int) -> None:
        er_tools.rng.seed(value)

    def make_types(self) -> None:
        er_creator.create_type("B_FIT", ErFitness, weights=(1.0, -1.0, 1.0))
        er_creator.create_type("B_IND", list, fitness=er_creator.B_FIT)
        er_creator.create_type("B_FIT_SO", ErFitness, weights=(-1.0,))
        er_creator.create_type("B_IND_SO", list, fitness=er_creator.B_FIT_SO)
        er_creator.create_type("B_FIT_GP", ErFitness, weights=(-1.0,))
        er_creator.create_type("B_FIT_LC", ErFitness, weights=tuple(-1.0 for _ in range(500)))
        er_creator.create_type("B_IND_LC", list, fitness=er_creator.B_FIT_LC)

    def drop_types(self) -> None:
        drop_creator_types(er_creator)

    def wrap_mo(self, genomes: list[list[float]], fits: list[tuple]) -> list:
        return _wrap(er_creator.B_IND, genomes, fits)

    def wrap_many_case(self, genomes: list[list[float]], fits: list[tuple]) -> list:
        return _wrap(er_creator.B_IND_LC, genomes, fits)

    def wrap_so(self, genomes: list[list[int]]) -> list:
        return _wrap_so(er_creator.B_IND_SO, genomes)

    def sel_spea(self, population: list, count: int) -> object:
        return er_tools.sel_spea_2(population, count)

    def sel_nsga2(self, population: list, count: int) -> object:
        return er_tools.sel_nsga_2(population, count)

    def sel_nsga3(self, population: list, count: int, refs: object) -> object:
        return er_tools.sel_nsga_3(population, count, refs)

    def sel_tournament(self, population: list, count: int, contestants: int) -> object:
        return er_tools.sel_tournament(population, count, contestants=contestants)

    def sel_lexicase(self, population: list, count: int) -> object:
        return er_tools.sel_lexicase(population, count)

    def new_pareto(self) -> Any:
        return er_tools.ParetoFront()

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

    def compile_one(self, tree: object, pset: object) -> Callable[[float], object]:
        return er_gp.compile_tree(tree, pset)

    def clear_compile_cache(self) -> None:
        module = sys.modules.get("deap_er.private.programming.compilers")
        cache = getattr(module, "_compile_cache", None)
        if cache is not None and hasattr(cache, "clear"):
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


def _wrap(cls: Any, genomes: list[list[float]], fits: list[tuple]) -> list:
    population = []
    for genome, fit in zip(genomes, fits, strict=True):
        individual = cls(genome)
        individual.fitness.values = fit
        population.append(individual)
    return population


def _wrap_so(cls: Any, genomes: list[list[int]]) -> list:
    population = []
    for genome in genomes:
        individual = cls(genome)
        individual.fitness.values = (float(sum(genome)),)
        population.append(individual)
    return population
