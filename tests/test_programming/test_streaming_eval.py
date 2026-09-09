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
from deap_er import Fitness, Toolbox, creator, gp, tools

STREAM_FIT = "STREAM_FIT"
STREAM_IND = "STREAM_IND"


def _vadd_tape():
    pset = gp.make_column_pset(["first", "second"])
    gp.add_numpy_primitives(pset)
    mapping = pset.mapping
    tree = gp.PrimitiveTree([mapping["vadd"], mapping["first"], mapping["second"]])
    return gp.lower_tree(tree, pset)


def _rolling_mean_tape(window=3):
    pset = gp.make_column_pset(["first"])
    gp.add_window_primitives(pset)
    gp.add_window_ephemeral(pset, "STREAM_WINDOW", window, window)
    mapping = pset.mapping
    leaf = pset.terminals[gp.Window][0]
    tree = gp.PrimitiveTree([mapping["rolling_mean"], mapping["first"], leaf()])
    return gp.lower_tree(tree, pset)


def _pack(*columns):
    return numpy.ascontiguousarray(numpy.stack(columns, axis=1))


def test_appended_rows_match_a_one_shot_full_matrix():
    prefix = _pack(numpy.arange(8.0))
    suffix = _pack(numpy.arange(8.0, 12.0))
    grown = numpy.vstack([prefix, suffix])
    tape = _rolling_mean_tape(3)

    actual = gp.interpret_tapes([tape], grown)
    expected = gp.rolling_mean(grown[:, 0], 3)

    numpy.testing.assert_allclose(actual[0], expected, equal_nan=True)
    prefix_rows = prefix.shape[0]
    numpy.testing.assert_allclose(actual[0, :prefix_rows], expected[:prefix_rows], equal_nan=True)
    numpy.testing.assert_allclose(actual[0, prefix_rows:], expected[prefix_rows:], equal_nan=True)


def test_suffix_without_history_misses_the_rolling_oracle():
    prefix = _pack(numpy.arange(8.0))
    suffix = _pack(numpy.arange(8.0, 12.0))
    grown = numpy.vstack([prefix, suffix])
    tape = _rolling_mean_tape(3)

    full = gp.interpret_tapes([tape], grown)
    suffix_only = gp.interpret_tapes([tape], suffix)

    assert not numpy.allclose(suffix_only, full[:, prefix.shape[0] :], equal_nan=True)


def test_prefix_scores_do_not_see_rows_that_have_not_arrived():
    prefix = _pack(numpy.arange(8.0))
    suffix = _pack(numpy.arange(8.0, 12.0))
    grown = numpy.vstack([prefix, suffix])
    tape = _rolling_mean_tape(3)

    before = gp.interpret_tapes([tape], prefix)
    after = gp.interpret_tapes([tape], grown)

    numpy.testing.assert_allclose(after[:, : prefix.shape[0]], before, equal_nan=True)


def test_append_requires_invalidating_stale_fitness():
    creator.create_type(STREAM_FIT, Fitness, weights=(-1.0,))
    creator.create_type(STREAM_IND, list, fitness=creator.__dict__[STREAM_FIT])
    try:
        tape = _vadd_tape()
        prefix = _pack(numpy.ones(4), numpy.zeros(4))
        suffix = _pack(numpy.full(4, 3.0), numpy.ones(4))
        target = numpy.zeros(8)
        individual = creator.__dict__[STREAM_IND]([0])

        def evaluate_batch(individuals):
            predicted = gp.interpret_tapes([tape] * len(individuals), matrix)
            rows = matrix.shape[0]
            return [
                tools.case_errors(predicted[index], target[:rows], [(0, rows)])
                for index in range(len(individuals))
            ]

        toolbox = Toolbox()
        toolbox.register("evaluate_batch", evaluate_batch)
        matrix = prefix
        tools.evaluate_invalid(toolbox, [individual])
        stale = individual.fitness.values
        matrix = numpy.vstack([prefix, suffix])
        assert individual.fitness.is_valid()
        assert individual.fitness.values == stale
        del individual.fitness.values
        tools.evaluate_invalid(toolbox, [individual])
        assert individual.fitness.values != stale
    finally:
        del creator.__dict__[STREAM_FIT]
        del creator.__dict__[STREAM_IND]
