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
"""deap-er-only columnar GP and semantic benches."""

from __future__ import annotations

import numpy
from deap_er import gp as er_gp
from deap_er import tools as er_tools

from .report import CaseResult
from .timing import mean_ms


def _column_pset(window_name: str) -> object:
    """Build a small columnar primitive set.

    Args:
        window_name: Ephemeral window constant name.

    Returns:
        A typed primitive set with NumPy and window kits.
    """
    pset = er_gp.make_column_pset(["first", "second"])
    er_gp.add_numpy_primitives(pset)
    er_gp.add_window_primitives(pset)
    er_gp.add_window_ephemeral(pset, window_name, 1, 6)
    return pset


def _trees(pset: object, count: int, seed: int) -> list:
    """Generate columnar trees.

    Args:
        pset: Columnar primitive set.
        count: Number of trees.
        seed: RNG seed.

    Returns:
        ``PrimitiveTree`` individuals.
    """
    er_tools.rng.seed(seed)
    return [er_gp.PrimitiveTree(er_gp.gen_half_and_half(pset, 1, 3)) for _ in range(count)]


def run_unique_columnar() -> list[CaseResult]:
    """Time columnar kits, tape backends, and semantic helpers.

    Returns:
        Unique columnar GP cases.
    """
    pset = _column_pset("BENCH_W")
    pair = er_gp.make_column_pset(["first", "second"])
    er_gp.add_numpy_primitives(pair)
    er_gp.add_pair_window_primitives(pair)
    er_gp.add_ts_primitives(pair)
    er_gp.add_window_ephemeral(pair, "BENCH_PAIR_W", 2, 6)
    trees = _trees(pset, 12, 81)
    tapes = []
    for tree in trees:
        try:
            tapes.append(er_gp.lower_tree(tree, pset))
        except ValueError:
            continue
    cols = (numpy.linspace(-1.0, 1.0, 48), numpy.sin(numpy.linspace(0.0, 6.0, 48)))
    matrix = numpy.ascontiguousarray(numpy.stack(cols, axis=1))
    predicted = numpy.sin(numpy.linspace(0.0, 8.0, 64))
    target = predicted + 0.05
    ranges = [(0, 16), (16, 32), (32, 48), (48, 64)]
    semantics = numpy.random.default_rng(82).normal(size=(24, 32))

    def columnar_gen() -> None:
        er_tools.rng.seed(85)
        for _ in range(16):
            er_gp.PrimitiveTree(er_gp.gen_half_and_half(pset, 1, 3))

    def opcode() -> None:
        for tree in trees:
            er_gp.compile_tree(tree, pset, backend="opcode")(*cols)

    def pair_ts() -> None:
        er_tools.rng.seed(86)
        tree = er_gp.PrimitiveTree(er_gp.gen_half_and_half(pair, 1, 3))
        try:
            tape = er_gp.lower_tree(tree, pair)
        except ValueError:
            return
        er_gp.interpret_tapes([tape], matrix)

    cases = [
        CaseResult(
            name="columnar generate x16",
            description="make_column_pset + numpy/window kits",
            shared=False,
            deap_er_ms=mean_ms(columnar_gen),
            feature="make_column_pset",
        ),
        CaseResult(
            name="compile_tree backend=opcode n=12",
            description="Opcode tape compile and eval",
            shared=False,
            deap_er_ms=mean_ms(opcode),
            feature="compile_tree opcode",
        ),
        CaseResult(
            name="interpret_tapes n=12 rows=48",
            description="Packed columnar tape batch",
            shared=False,
            deap_er_ms=mean_ms(lambda: er_gp.interpret_tapes(tapes, matrix)),
            feature="interpret_tapes",
        ),
        CaseResult(
            name="pair/ts primitives lower+interpret",
            description="add_pair_window_primitives + add_ts_primitives",
            shared=False,
            deap_er_ms=mean_ms(pair_ts),
            feature="add_pair_window_primitives",
        ),
        CaseResult(
            name="case_errors 4 segments n=64",
            description="Segment MSE helper",
            shared=False,
            deap_er_ms=mean_ms(lambda: er_tools.case_errors(predicted, target, ranges)),
            feature="case_errors",
        ),
        CaseResult(
            name="semantic_moments + nearest n=24",
            description="Semantic descriptors and neighbors",
            shared=False,
            deap_er_ms=mean_ms(_semantic_run(semantics)),
            feature="semantic_moments",
        ),
    ]
    if er_gp.numba_available() and tapes:
        er_gp.warmup_numba()
        cases.append(
            CaseResult(
                name="interpret_tapes backend=numba n=12",
                description="Numba tape batch interpreter",
                shared=False,
                deap_er_ms=mean_ms(
                    lambda: er_gp.interpret_tapes(tapes, matrix, backend="numba"),
                    repeat=10,
                    warmup=1,
                ),
                feature="compile_tree numba",
                notes="repeat=10, warmup=1 after warmup_numba()",
                repeat=10,
                warmup=1,
            )
        )
    return cases


def _semantic_run(semantics: numpy.ndarray):
    """Return a timed callable for semantic moments and nearest neighbors.

    Args:
        semantics: Packed ``(n_individuals, n_rows)`` matrix.

    Returns:
        Nullary callable.
    """

    def run() -> None:
        desc = er_tools.semantic_moments(semantics)
        er_tools.semantic_nearest(desc[0], desc, k=5)

    return run
