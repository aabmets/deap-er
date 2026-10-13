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
import hashlib

import numpy
from deap_er import gp, tools

__all__ = [
    "GOLDEN_DIGEST",
    "GOLDEN_TEXTS",
    "MATH_TEXTS",
    "STRIDES",
    "digest",
    "precision_matrix",
    "precision_pset",
    "random_tapes",
    "text_tapes",
]

STRIDES = {"s": 3}

# sha256 of interpret_tapes over GOLDEN_TEXTS and precision_matrix(), as
# deap-er 3.2.9 computed it before the precision argument existed. The
# opcode backend and the serial and parallel Numba paths all give it.
GOLDEN_DIGEST = "8405ee329405d86295cafd5c497a42eb6488bfe6c9a75c94386c4c441ba13154"

# Only correctly rounded arithmetic, comparisons, and window statistics,
# so the pinned digests do not depend on the platform's libm or SIMD.
GOLDEN_TEXTS = (
    "vadd(vmul(x, y), vsub(s, x))",
    "vdiv(x, y)",
    "vsqrt(vabs(y))",
    "vneg(vdiv(s, vsub(x, x)))",
    "vwhere(vgt(x, y), x, vneg(y))",
    "vwhere(vand(vge(x, s), vnot(vlt(y, x))), s, y)",
    "vwhere(vor(veq(x, s), vle(y, s)), x, y)",
    "rolling_sum(x, 5)",
    "rolling_mean(s, 2)",
    "rolling_std(y, 4)",
    "rolling_min(s, 3)",
    "rolling_max(x, 6)",
    "ema(y, 4)",
    "delay(s, 2)",
    "diff(x, 3)",
    "rolling_corr(x, s, 3)",
    "rolling_cov(x, y, 5)",
    "rolling_beta(y, x, 4)",
    "ts_rank(x, 5)",
    "ts_argmax(s, 2)",
    "ts_argmin(y, 6)",
    "rolling_mean(vmul(x, s), 3)",
    "diff(rolling_std(s, 2), 1)",
)

# Transcendental primitives, compared only within one process.
MATH_TEXTS = (
    "vlog(vabs(x))",
    "vsin(vmul(x, y))",
    "vcos(rolling_mean(s, 4))",
)


def precision_pset():
    return gp.columnar_pset(
        ["x", "y", "s"], window=(1, 6), ts=True, pair_windows=True, strides=STRIDES
    )


def precision_matrix(rows=96, pad=7, seed=41):
    """Return a float32 matrix with NaN padding, a zero, and a held column."""
    generator = numpy.random.default_rng(seed)
    held = numpy.repeat(generator.normal(size=rows // 3 + 1), 3)[:rows]
    matrix = numpy.stack(
        [generator.normal(size=rows), generator.normal(size=rows) * 2.0, held], axis=1
    ).astype(numpy.float32)
    matrix[:pad] = numpy.nan
    matrix[pad + 5, 1] = 0.0
    matrix[pad + 20, 1] = numpy.nan
    return numpy.ascontiguousarray(matrix)


def text_tapes(pset, texts):
    return [gp.lower_tree(gp.PrimitiveTree.from_string(text, pset), pset) for text in texts]


def random_tapes(pset, count, seed=53):
    tools.rng.seed(seed)
    tapes = []
    while len(tapes) < count:
        tree = gp.PrimitiveTree(gp.gen_half_and_half(pset, 2, 5))
        try:
            tapes.append(gp.lower_tree(tree, pset))
        except ValueError:
            continue
    return tapes


def digest(results):
    """Hash results bit for bit, with every NaN in one canonical form."""
    canonical = numpy.where(numpy.isnan(results), numpy.nan, results)
    return hashlib.sha256(numpy.ascontiguousarray(canonical).tobytes()).hexdigest()
