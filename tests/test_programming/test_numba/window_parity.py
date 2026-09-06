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
from deap_er import gp

__all__ = [
    "COLUMNS",
    "UNARY",
    "PAIR",
    "TS_OPS",
    "NAN",
    "INF",
    "window_kit",
    "window_span",
    "window_tree",
    "check_parity",
]

COLUMNS = ["first", "second", "third"]
UNARY = ["rolling_sum", "rolling_mean", "rolling_std", "rolling_min", "rolling_max"]
PAIR = ["rolling_corr", "rolling_cov", "rolling_beta"]
TS_OPS = ["ts_rank", "ts_argmax", "ts_argmin"]
NAN = numpy.nan
INF = numpy.inf


def window_kit(window, kind):
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    if kind == "pair":
        gp.add_pair_window_primitives(pset)
    elif kind == "ts":
        gp.add_ts_primitives(pset)
    else:
        gp.add_window_primitives(pset)
    gp.add_window_ephemeral(pset, f"INC_W_{window}", window, window)
    return pset


def window_span(pset):
    return pset.terminals[gp.Window][0]()


def window_tree(pset, name, kind):
    mapping = pset.mapping
    span = window_span(pset)
    if kind == "pair":
        nodes = [mapping[name], mapping["first"], mapping["second"], span]
    else:
        nodes = [mapping[name], mapping["first"], span]
    return gp.PrimitiveTree(nodes)


def _columns(first, second=None):
    first = numpy.asarray(first, dtype=numpy.float64)
    if second is None:
        second = first * 0.5 + 1.0
    third = numpy.abs(first) + 1.0
    return first, numpy.asarray(second, dtype=numpy.float64), third


def check_parity(tree, pset, columns, atol=1e-12):
    packed = _columns(*columns) if len(columns) < 3 else columns
    expected = numpy.broadcast_to(
        numpy.asarray(gp.compile_tree(tree, pset)(*packed), dtype=numpy.float64),
        (packed[0].shape[0],),
    )
    actual = gp.compile_tree(tree, pset, backend="numba")(*packed)
    numpy.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-9, atol=atol)
