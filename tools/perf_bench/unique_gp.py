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
"""deap-er-only genetic-programming benches."""

from __future__ import annotations

import math
import operator

from deap_er import Toolbox
from deap_er import creator as er_creator
from deap_er import gp as er_gp
from deap_er import tools as er_tools

from .data import so_data, wrap_so
from .report import CaseResult
from .timing import mean_ms


def run_unique_gp() -> list[CaseResult]:
    """Time leaf-only generate, weights, SlimGP, promote, and batch eval.

    Returns:
        Unique GP cases.
    """
    weighted = er_gp.PrimitiveSet("MAIN", 1)
    weighted.add_primitive(operator.add, 2, name="add", weight=4.0)
    weighted.add_primitive(operator.mul, 2, name="mul", weight=1.0)
    weighted.add_primitive(operator.neg, 1, name="neg", weight=1.0)
    leaf = er_gp.PrimitiveSetTyped("LEAF", [], int)
    leaf.add_terminal(3, int, name="three")

    def leaf_only() -> None:
        for _ in range(40):
            er_gp.generate(leaf, 0, 0, lambda _h, _d: True, int)

    def weighted_gen() -> None:
        er_tools.rng.seed(84)
        for _ in range(30):
            er_gp.gen_half_and_half(weighted, 1, 3)

    def promote() -> None:
        typed = er_gp.PrimitiveSetTyped("PROMO", [float, float], float)
        typed.add_primitive(operator.add, [float, float], float)
        typed.add_primitive(operator.mul, [float, float], float)
        tree = er_gp.PrimitiveTree.from_string("add(ARG0, mul(ARG1, ARG0))", typed)
        er_gp.promote_subtree(typed, tree)
        er_gp.gen_full(typed, 1, 2)

    def slim() -> None:
        er_tools.rng.seed(87)
        slim_set = er_gp.PrimitiveSet("SLIM", 1)
        slim_set.add_primitive(operator.add, 2)
        slim_set.add_primitive(operator.sub, 2)
        slim_set.add_primitive(operator.mul, 2)
        slim_set.add_primitive(lambda x: 1 / (1 + math.exp(-x)), 1, name="lf")
        head = er_gp.PrimitiveTree(er_gp.gen_grow(slim_set, 1, 2))
        individual = er_gp.SlimTree.from_tree(head)
        er_gp.mut_slim(individual, slim_set, inflate_prob=0.8, min_depth=1, max_depth=2)
        er_gp.compile_slim_tree(individual, slim_set)(0.5)

    def tune() -> None:
        er_tools.rng.seed(88)
        eph = er_gp.PrimitiveSet("TUNE", 1)
        eph.add_primitive(operator.add, 2)
        eph.add_ephemeral_constant("BENCH_EPH", lambda: 0.25)
        node = eph.terminals[object][-1]
        tree = er_creator.U_IND_GP([eph.mapping["add"], node(), eph.mapping["ARG0"]])
        strategy = er_tools.Strategy(centroid=[0.25], sigma=0.2, offsprings=4)

        def evaluate(individual: object) -> tuple[float]:
            func = er_gp.compile_tree(individual, eph)
            err = func(0.0) - 1.0
            return (err * err,)

        er_gp.tune_ephemerals(tree, strategy, evaluate, n_gen=2)

    def batch() -> None:
        toolbox = Toolbox()
        toolbox.register("mate", er_tools.cx_two_point)
        toolbox.register("mutate", er_tools.mut_flip_bit, mut_prob=0.2)
        toolbox.register("select", er_tools.sel_tournament, contestants=3)
        toolbox.register("evaluate", lambda individual: (float(sum(individual)),))
        toolbox.register(
            "evaluate_batch",
            lambda individuals: [(float(sum(ind)),) for ind in individuals],
        )
        er_tools.rng.seed(89)
        er_tools.ea_simple(toolbox, wrap_so(so_data(20, 16, 89)), 4, 0.5, 0.2)

    return [
        CaseResult(
            name="generate leaf-only type x40",
            description="generate() closes a terminal-only type",
            shared=False,
            deap_er_ms=mean_ms(leaf_only),
            feature="leaf-only generate()",
        ),
        CaseResult(
            name="weighted primitive generate x30",
            description="add_primitive(..., weight=) sampling",
            shared=False,
            deap_er_ms=mean_ms(weighted_gen),
            feature="weighted primitives",
        ),
        CaseResult(
            name="promote_subtree + gen_full",
            description="Lift a typed subtree into the set",
            shared=False,
            deap_er_ms=mean_ms(promote),
            feature="promote_subtree",
        ),
        CaseResult(
            name="SlimTree mut_slim + compile",
            description="SlimGP inflate/deflate and compile",
            shared=False,
            deap_er_ms=mean_ms(slim),
            feature="SlimTree",
        ),
        CaseResult(
            name="tune_ephemerals n_gen=2",
            description="Memetic boxed CMA on ephemeral leaves",
            shared=False,
            deap_er_ms=mean_ms(tune, repeat=10, warmup=1),
            feature="tune_ephemerals",
            notes="repeat=10, warmup=1; n_gen=2, dim=1",
            repeat=10,
            warmup=1,
        ),
        CaseResult(
            name="ea_simple evaluate_batch n=20 gens=4",
            description="Algorithm batch evaluation path",
            shared=False,
            deap_er_ms=mean_ms(batch, repeat=20, warmup=2),
            feature="evaluate_batch",
            notes="repeat=20",
            repeat=20,
            warmup=2,
        ),
    ]
