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
import operator
import random
from typing import Any

import numpy
from deap_er import gp


def seeded_population(cls: Any, seed: int, count: int, genes: int, objectives: int) -> list[Any]:
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


def selection_indices(population: list[Any]) -> list[int]:
    """Return the origin index of every individual in a selection."""
    return [int(ind.idx) for ind in population]


def bit_strings(population: list[Any]) -> list[str]:
    """Return a binary population as one string per individual."""
    return ["".join(str(int(gene)) for gene in ind) for ind in population]


def flatten_values(*values: Any) -> list[float]:
    """Flatten nested numeric values into one list of floats."""
    out: list[float] = []
    for value in values:
        if isinstance(value, int | float | numpy.floating | numpy.integer):
            out.append(float(value))
        elif isinstance(value, numpy.ndarray):
            out.extend(float(v) for v in value.ravel())
        else:
            for item in value:
                out.extend(flatten_values(item))
    return out


def untyped_pset() -> gp.PrimitiveSet:
    """Build the untyped primitive set used by the crossover cases."""
    pset = gp.PrimitiveSet("MAIN", 2)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.mul, 2)
    pset.add_primitive(operator.sub, 2)
    pset.add_primitive(operator.neg, 1)
    return pset


def typed_pset() -> gp.PrimitiveSetTyped:
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
