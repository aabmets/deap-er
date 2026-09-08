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
"""Shared genomes, GP source, and creator-type helpers for the bench."""

from __future__ import annotations

import random
from typing import Any

from deap_er import Fitness as ErFitness
from deap_er import creator as er_creator
from deap_er import gp as er_gp

TYPE_NAMES = (
    "B_FIT",
    "B_IND",
    "B_FIT_SO",
    "B_IND_SO",
    "B_FIT_GP",
    "B_FIT_LC",
    "B_IND_LC",
    "U_FIT",
    "U_IND",
    "U_FIT_SO",
    "U_IND_SO",
    "U_FIT_LC",
    "U_IND_LC",
    "U_FIT_ME",
    "U_IND_ME",
    "U_IND_GP",
)
_GP_PRIMS = (("add", 2), ("mul", 2), ("neg", 1))
_GP_TERM = "ARG0"
_GP_TERM_RATIO = 1 / (1 + len(_GP_PRIMS))


def drop_creator_types(module: Any) -> None:
    """Remove bench types from a creator module.

    Args:
        module: ``creator`` module that may hold the bench types.
    """
    for name in TYPE_NAMES:
        if name in module.__dict__:
            delattr(module, name)


def make_unique_types() -> None:
    """Create deap-er types used by unique feature benches."""
    er_creator.create_type("U_FIT", ErFitness, weights=(1.0, -1.0, 1.0))
    er_creator.create_type("U_IND", list, fitness=er_creator.U_FIT)
    er_creator.create_type("U_FIT_SO", ErFitness, weights=(-1.0,))
    er_creator.create_type("U_IND_SO", list, fitness=er_creator.U_FIT_SO)
    er_creator.create_type("U_FIT_LC", ErFitness, weights=tuple(-1.0 for _ in range(80)))
    er_creator.create_type("U_IND_LC", list, fitness=er_creator.U_FIT_LC)
    er_creator.create_type("U_FIT_ME", ErFitness, weights=(1.0,))
    er_creator.create_type("U_IND_ME", list, fitness=er_creator.U_FIT_ME)
    er_creator.create_type("U_IND_GP", er_gp.PrimitiveTree, fitness=er_creator.U_FIT_SO)


def wrap_mo(genomes: list[list[float]], fits: list[tuple], *, many: bool = False) -> list:
    """Wrap genomes as deap-er multi-objective individuals.

    Args:
        genomes: Gene lists.
        fits: Matching fitness tuples.
        many: If True, use the many-case lexicase type.

    Returns:
        Evaluated individuals.
    """
    cls = er_creator.U_IND_LC if many else er_creator.U_IND
    population = []
    for genome, fit in zip(genomes, fits, strict=True):
        individual = cls(genome)
        individual.fitness.values = fit
        population.append(individual)
    return population


def wrap_so(genomes: list[list[int]], *, map_elites: bool = False) -> list:
    """Wrap genomes as single-objective deap-er individuals.

    Args:
        genomes: Gene lists.
        map_elites: If True, use the MAP-Elites maximize type.

    Returns:
        Evaluated individuals.
    """
    cls = er_creator.U_IND_ME if map_elites else er_creator.U_IND_SO
    population = []
    for genome in genomes:
        individual = cls(genome)
        individual.fitness.values = (float(sum(genome)),)
        population.append(individual)
    return population


def mo_data(
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


def so_data(count: int, genes: int, seed: int) -> list[list[int]]:
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


def optimal_front(count: int, objectives: int, seed: int) -> list[tuple]:
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


def _gp_node(rng: random.Random, height: int, depth: int, min_depth: int, grow: bool) -> str:
    """Build one shared GP expression node.

    Args:
        rng: Dedicated Python RNG, independent of either library.
        height: Drawn tree height.
        depth: Depth of this node.
        min_depth: Earliest depth a terminal is allowed.
        grow: If True, use grow (mixed leaf depths); else full.

    Returns:
        A Python expression using ``add``, ``mul``, ``neg``, and ``ARG0``.
    """
    at_height = depth == height
    early_term = grow and depth >= min_depth and rng.random() < _GP_TERM_RATIO
    if at_height or early_term:
        return _GP_TERM
    name, arity = rng.choice(_GP_PRIMS)
    args = ", ".join(_gp_node(rng, height, depth + 1, min_depth, grow) for _ in range(arity))
    return f"{name}({args})"


def gp_exprs(count: int, seed: int) -> list[str]:
    """Build shared half-and-half expression strings for compile benches.

    Args:
        count: Number of expressions.
        seed: RNG seed.

    Returns:
        Expression source that both libraries compile as-is.
    """
    rng = random.Random(seed)
    exprs = []
    for _ in range(count):
        grow = rng.choice((True, False))
        height = rng.randint(1, 4)
        exprs.append(_gp_node(rng, height, 0, 1, grow))
    return exprs
