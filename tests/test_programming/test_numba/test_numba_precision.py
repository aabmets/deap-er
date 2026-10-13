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
import subprocess
import sys
import tracemalloc
from typing import Any

import numpy
import pytest
from deap_er import gp
from deap_er.private.programming.numba.numba_batch import batch_kernel
from deap_er.private.programming.numba.numba_compile import build
from tests.test_programming.precision_cases import (
    GOLDEN_DIGEST,
    GOLDEN_TEXTS,
    MATH_TEXTS,
    digest,
    precision_matrix,
    precision_pset,
    random_tapes,
    text_tapes,
)

pytestmark = pytest.mark.skipif(
    not gp.numba_available(), reason="the optional numba extra is not installed"
)

# Typed as Any so the type checker lets the bad value reach the check.
BAD_PRECISION: Any = "float16"
PARALLEL = [False, True]


def _matrix_dtypes(kernel, position):
    """Return the dtype of the matrix argument of each compiled signature."""
    return {str(signature[position].dtype) for signature in kernel.signatures}


@pytest.mark.parametrize("parallel", PARALLEL)
@pytest.mark.parametrize("cast", [False, True])
def test_the_default_precision_reproduces_the_3_2_9_results(parallel, cast):
    pset = precision_pset()
    m32 = precision_matrix()
    matrix = m32 if cast else m32.astype(numpy.float64)
    tapes = text_tapes(pset, GOLDEN_TEXTS)
    results = gp.interpret_tapes(tapes, matrix, backend="numba", parallel=parallel)
    assert digest(results) == GOLDEN_DIGEST


@pytest.mark.parametrize("parallel", PARALLEL)
def test_float32_storage_is_bit_identical_on_every_opcode(parallel):
    pset = precision_pset()
    tapes = text_tapes(pset, GOLDEN_TEXTS + MATH_TEXTS) + random_tapes(pset, 40)
    m32 = precision_matrix()
    expected = gp.interpret_tapes(
        tapes, m32.astype(numpy.float64), backend="numba", parallel=parallel
    )
    actual = gp.interpret_tapes(tapes, m32, backend="numba", parallel=parallel, precision="float32")
    assert actual.dtype == numpy.float64
    assert actual.shape == (len(tapes), m32.shape[0])
    assert numpy.array_equal(actual, expected, equal_nan=True)


@pytest.mark.parametrize("parallel", PARALLEL)
def test_a_float32_matrix_is_not_copied(parallel):
    names = [f"c{index}" for index in range(64)]
    pset = gp.columnar_pset(names, window=None)
    tapes = text_tapes(pset, ["vadd(c0, c1)", "rolling_mean(vmul(c2, c0), 4)"])
    m32 = numpy.random.default_rng(5).normal(size=(100_000, 64)).astype(numpy.float32)
    run = lambda: gp.interpret_tapes(  # noqa: E731
        tapes, m32, backend="numba", parallel=parallel, precision="float32"
    )
    run()
    tracemalloc.start()
    try:
        run()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # Stacks and the result of a few float64 rows fit well below a
    # float64 copy of the 64-column matrix.
    assert peak < m32.size * 8 // 2


def test_bind_tape_runs_a_float32_matrix():
    pset = precision_pset()
    m32 = precision_matrix()
    for tape in text_tapes(pset, GOLDEN_TEXTS):
        expected = gp.bind_tape(tape)(m32.astype(numpy.float64))
        actual = gp.bind_tape(tape, precision="float32")(m32)
        assert actual.dtype == numpy.float64
        assert numpy.array_equal(actual, expected, equal_nan=True)


def test_a_dispatch_kernel_rejects_float32_storage():
    pset = precision_pset()
    tapes = text_tapes(pset, GOLDEN_TEXTS[:1])
    m32 = precision_matrix()
    dispatch = object()
    with pytest.raises(ValueError, match="cannot be combined with a dispatch kernel"):
        gp.interpret_tapes(tapes, m32, backend="numba", dispatch=dispatch, precision="float32")
    with pytest.raises(ValueError, match="cannot be combined with a dispatch kernel"):
        gp.bind_tape(tapes[0], dispatch, precision="float32")
    with pytest.raises(ValueError, match="cannot be combined with a dispatch kernel"):
        gp.warmup_numba(dispatch=dispatch, precisions=("float64", "float32"))


@pytest.mark.parametrize("precisions", [("float16",), ("float64", "f4"), "float32"])
def test_warmup_numba_rejects_an_unknown_precision(precisions):
    tape = text_tapes(precision_pset(), GOLDEN_TEXTS[:1])[0]
    with pytest.raises(ValueError, match="precision must be 'float64' or 'float32'"):
        gp.warmup_numba(precisions=precisions)
    with pytest.raises(ValueError, match="precision must be 'float64' or 'float32'"):
        gp.bind_tape(tape, precision=BAD_PRECISION)


_CACHE_PROBE = """
import numpy
from deap_er import gp
from deap_er.private.programming.numba.numba_batch import batch_kernel
from deap_er.private.programming.numba.numba_compile import build
pset = gp.make_column_pset(["first"])
gp.add_numpy_primitives(pset)
tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["first"]]), pset)
matrix = numpy.zeros((2, 1), dtype=numpy.float32)
gp.bind_tape(tape, precision="float32")(matrix)
gp.interpret_tapes([tape], matrix, backend="numba", precision="float32")
for kernel in (build(), batch_kernel(False, False)):
    stats = kernel.stats
    print(sum(stats.cache_hits.values()), sum(stats.cache_misses.values()))
"""


def test_warmup_numba_compiles_both_precisions_into_the_disk_cache():
    gp.warmup_numba(parallel=True, precisions=("float64", "float32"))

    assert _matrix_dtypes(build(), 3) == {"float64", "float32"}
    for parallel in PARALLEL:
        assert _matrix_dtypes(batch_kernel(parallel, False), 2) == {"float64", "float32"}
    completed = subprocess.run(
        [sys.executable, "-c", _CACHE_PROBE], capture_output=True, text=True, timeout=600
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["1", "0", "1", "0"]
