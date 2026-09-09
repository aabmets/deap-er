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
import numpy
import pytest
from deap_er import Fitness, Toolbox, creator, gp, tools

COL_FIT = "COL_SETUP_FIT"
COL_IND = "COL_SETUP_IND"


def test_columnar_pset_registers_the_standard_kits():
    pset = gp.columnar_pset(["level", "flow"], window=(2, 8), window_name="kit_window")

    assert pset.arguments == ["level", "flow"]
    assert "vadd" in pset.mapping
    assert "delay" in pset.mapping
    assert pset.terminals[gp.Window]
    assert "rolling_corr" not in pset.mapping
    assert "ts_rank" not in pset.mapping


def test_columnar_pset_opts_into_pair_and_ts_kits():
    pset = gp.columnar_pset(
        ["level", "flow"],
        window=None,
        pair_windows=True,
        ts=True,
    )

    assert "rolling_beta" in pset.mapping
    assert "ts_rank" in pset.mapping
    assert not pset.terminals.get(gp.Window)


def test_evaluate_columnar_scores_unique_trees_once():
    pset = gp.columnar_pset(["level"], window=None)
    level = numpy.linspace(0.0, 1.0, 8)
    target = gp.vabs(level)
    tree = gp.PrimitiveTree.from_string("vabs(level)", pset)
    other = gp.PrimitiveTree.from_string("vneg(level)", pset)
    matrix = numpy.column_stack([level])

    scores = gp.evaluate_columnar([tree, tree, other], pset, matrix, target, min_valid=1)

    assert scores[0] == scores[1]
    assert scores[0][0] == pytest.approx(0.0)
    assert scores[2][0] > scores[0][0]


def test_evaluate_columnar_uses_case_errors_and_rejects_bad_target():
    pset = gp.columnar_pset(["level"], window=None)
    level = numpy.arange(8, dtype=numpy.float64)
    target = level.copy()
    tree = gp.PrimitiveTree.from_string("level", pset)
    matrix = numpy.column_stack([level])

    reduced = gp.evaluate_columnar([tree], pset, matrix, target, cases=[(0, 4), (4, 8)])
    full = gp.evaluate_columnar([tree], pset, matrix, target, cases=[(0, 4), (4, 8)], reduce=False)
    assert reduced == [(0.0,)]
    assert full == [(0.0, 0.0)]
    target_with_nan = level.copy()
    target_with_nan[:4] = numpy.nan
    mixed = gp.evaluate_columnar(
        [tree],
        pset,
        matrix,
        target_with_nan,
        cases=[(0, 4), (4, 8)],
        empty=float("inf"),
    )
    assert mixed[0][0] == float("inf")
    assert gp.evaluate_columnar([], pset, matrix, target) == []
    assert gp.evaluate_columnar([tree], pset, matrix, target, cases=[], empty=1.5e6) == [(1.5e6,)]
    assert gp.evaluate_columnar([tree], pset, matrix, target, cases=[], reduce=False) == [()]
    with pytest.raises(ValueError, match="one-dimensional"):
        gp.evaluate_columnar([tree], pset, matrix, numpy.ones((8, 1)))


def test_register_gp_plus_evaluate_columnar_runs_ea_simple():
    creator.create_type(COL_FIT, Fitness, weights=(-1.0,))
    creator.create_type(COL_IND, gp.PrimitiveTree, fitness=creator.__dict__[COL_FIT])
    try:
        pset = gp.columnar_pset(["level"], window=(2, 4), window_name="col_setup_win")
        level = numpy.linspace(-1.0, 1.0, 16)
        target = gp.vabs(level)
        matrix = numpy.column_stack([level])
        toolbox = Toolbox()
        gp.register_gp(
            toolbox,
            pset,
            individual=creator.__dict__[COL_IND],
            max_depth=2,
            height_limit=4,
            backend="opcode",
        )
        toolbox.register(
            "evaluate_batch",
            gp.evaluate_columnar,
            pset=pset,
            matrix=matrix,
            target=target,
            min_valid=1,
        )
        population = toolbox.population(size=8)
        _, logbook = tools.ea_simple(toolbox, population, generations=1, cx_prob=0.5, mut_prob=0.2)
        assert logbook.select("gen") == [0, 1]
        assert all(ind.fitness.is_valid() for ind in population)
    finally:
        del creator.__dict__[COL_FIT]
        del creator.__dict__[COL_IND]
