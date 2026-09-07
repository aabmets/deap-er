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
from deap_er import gp

__all__ = [
    "BATCH_TRIPLE",
    "OPS_TRIPLE",
    "SCRATCH_USER",
    "consumer_dispatch",
]

BATCH_TRIPLE = gp.USER_BASE + 81
OPS_TRIPLE = gp.USER_BASE + 61
SCRATCH_USER = gp.USER_BASE + 63
_DISPATCH = None


def consumer_dispatch():
    global _DISPATCH
    if _DISPATCH is not None:
        return _DISPATCH
    import numba  # optional extra, imported only when a consumer kernel runs

    @numba.njit(cache=True, nogil=True, error_model="numpy")
    def dispatch(op, sp, stack, columns, constants, scratch):
        if op in (BATCH_TRIPLE, OPS_TRIPLE):
            for index in range(columns.shape[0]):
                stack[sp - 1, index] = 3.0 * stack[sp - 1, index]
            return sp
        if op == SCRATCH_USER:
            for index in range(columns.shape[0]):
                stack[sp, index] = 3.0 * stack[sp - 1, index]
            for index in range(columns.shape[0]):
                stack[sp - 1, index] = stack[sp, index]
            return sp
        return -1

    _DISPATCH = dispatch
    return dispatch
