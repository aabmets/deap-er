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
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy
import pytest
from deap_er import gp

pytestmark = pytest.mark.skipif(
    not gp.numba_available(), reason="the optional numba extra is not installed"
)

THREADS = 4
ROUNDS = 6
TEXTS = [
    "rolling_mean(vmul(first, second), 7)",
    "rolling_std(vadd(first, rolling_sum(second, 3)), 5)",
    "vsub(rolling_max(first, 9), rolling_min(second, 4))",
]


def _tapes():
    pset = gp.make_column_pset(["first", "second"])
    gp.add_numpy_primitives(pset)
    gp.add_window_primitives(pset)
    trees = [gp.PrimitiveTree.from_string(text, pset) for text in TEXTS]
    return [gp.lower_tree(tree, pset) for tree in trees]


def test_serial_interpret_tapes_is_safe_from_several_threads():
    # The compiled kernels release the GIL, so threads that shared one
    # workspace overwrote each other's stack rows mid-tape.
    tapes = _tapes()
    matrices = [
        numpy.random.default_rng(seed).normal(size=(20_000, 2)) * (seed + 1)
        for seed in range(THREADS)
    ]
    expected = [gp.interpret_tapes(tapes, matrix, backend="numba") for matrix in matrices]
    barrier = threading.Barrier(THREADS)

    def worker(index):
        barrier.wait()
        wrong = 0
        for _ in range(ROUNDS):
            actual = gp.interpret_tapes(tapes, matrices[index], backend="numba")
            wrong += not numpy.array_equal(actual, expected[index], equal_nan=True)
        return wrong

    with ThreadPoolExecutor(THREADS) as pool:
        wrong = list(pool.map(worker, range(THREADS)))

    assert wrong == [0] * THREADS
