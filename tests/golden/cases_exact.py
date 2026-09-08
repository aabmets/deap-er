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

from deap_er import Toolbox, gp, tools
from tests.golden.case_support import (
    bit_strings,
    seeded_population,
    selection_indices,
    typed_pset,
    untyped_pset,
)


def selection_case_exact(types: SimpleNamespace, seed: int) -> dict[str, Any]:
    """SPEA-II and lexicase selection results for one seed."""
    out: dict[str, Any] = {}

    for sel_count in (2, 4, 7, 10, 12):
        population = seeded_population(types.two_obj, seed, 10, 3, 2)
        tools.rng.seed(seed + 300000)
        out[f"spea2/{sel_count}"] = selection_indices(tools.sel_spea_2(population, sel_count))

    population = seeded_population(types.four_obj, seed, 8, 3, 4)
    for label, args in (
        ("lexicase", ()),
        ("epsilon_auto", (None,)),
        ("epsilon_zero", (0.0,)),
        ("epsilon_half", (0.5,)),
    ):
        tools.rng.seed(seed + 200000)
        if label == "lexicase":
            chosen = tools.sel_lexicase(population, 6)
        else:
            chosen = tools.sel_epsilon_lexicase(population, 6, *args)
        out[label] = selection_indices(chosen)

    return out


def gp_crossover_case_exact(types: SimpleNamespace, seed: int) -> dict[str, Any]:
    """One-point and leaf-biased GP crossover results for one seed."""
    out: dict[str, Any] = {}

    for name, pset in (
        ("untyped", untyped_pset()),
        ("typed", typed_pset()),
    ):
        tools.rng.seed(seed)
        first = types.gp_ind(gp.gen_full(pset, 1, 4))
        second = types.gp_ind(gp.gen_grow(pset, 1, 4))
        tools.rng.seed(seed + 100000)
        gp.cx_one_point(first, second)
        out[f"one_point/{name}"] = [str(first), str(second)]

        tools.rng.seed(seed)
        third = types.gp_ind(gp.gen_full(pset, 1, 4))
        fourth = types.gp_ind(gp.gen_grow(pset, 1, 4))
        tools.rng.seed(seed + 100000)
        gp.cx_one_point_leaf_biased(third, fourth, 0.1 + (seed % 9) / 10.0)
        out[f"leaf_biased/{name}"] = [str(third), str(fourth)]

    return out


def _ea_toolbox() -> Toolbox:
    """Build a toolbox whose operators are all discrete."""
    toolbox = Toolbox()
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

        tools.rng.seed(seed + 600000)
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

        out[f"{name}/population"] = bit_strings(final)
        out[f"{name}/logbook"] = [dict(sorted(record.items())) for record in logbook]
        out[f"{name}/hof"] = bit_strings(list(hof))

    return out


def operators_case_exact(types: SimpleNamespace, seed: int) -> dict[str, Any]:
    """Integer mutation results for one seed."""
    tools.rng.seed(seed + 9000)
    size = 3 + seed % 6
    ind = types.min_ind([tools.rng.randint(0, 5) for _ in range(size)])
    tools.mut_uniform_int(ind, 0, 9, 0.5)
    return {"uniform_int": [int(gene) for gene in ind]}


def benchmarks_case_exact(types: SimpleNamespace, seed: int) -> dict[str, Any]:
    """ZDT1 and ZDT2 values, which use only sqrt and basic arithmetic."""
    out: dict[str, Any] = {}
    random.seed(seed)
    for dim in (2, 3, 5, 10):
        vector = types.min_ind(random.uniform(0.01, 1.0) for _ in range(dim))
        out[f"zdt1/{dim}"] = list(tools.bm_zdt_1(vector))
        out[f"zdt2/{dim}"] = list(tools.bm_zdt_2(vector))
    return out
