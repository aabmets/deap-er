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
from .private.programming.columnar import Array, Mask, Window, make_column_pset
from .private.programming.compilers import (
    build_tree_graph,
    compile_adf_tree,
    compile_tree,
    static_limit,
)
from .private.programming.crossover import cx_one_point, cx_one_point_leaf_biased
from .private.programming.generators import gen_full, gen_grow, gen_half_and_half, generate
from .private.programming.harm.harm import harm
from .private.programming.infix import tree_to_infix
from .private.programming.mutation import (
    mut_ephemeral,
    mut_insert,
    mut_node_replacement,
    mut_shrink,
    mut_uniform,
)
from .private.programming.numba.numba_ops import USER_DISPATCH_SIGNATURE, bind_tape, numba_available
from .private.programming.numpy.numpy_arith import (
    vabs,
    vadd,
    vcos,
    vdiv,
    vlog,
    vmul,
    vneg,
    vsin,
    vsqrt,
    vsub,
)
from .private.programming.numpy.numpy_logic import (
    vand,
    veq,
    vge,
    vgt,
    vle,
    vlt,
    vnot,
    vor,
    vwhere,
)
from .private.programming.numpy.numpy_ops import add_numpy_primitives, infer_fill
from .private.programming.opcodes import (
    BUILTIN_OPCODES,
    USER_BASE,
    Opcode,
    Tape,
    bind_numba_opcode,
    interpret_tape,
    lower_tree,
    numba_opcodes,
)
from .private.programming.primitives.primitive_nodes import Ephemeral, Primitive, Terminal
from .private.programming.primitives.primitive_set import PrimitiveSet
from .private.programming.primitives.primitive_set_typed import PrimitiveSetTyped
from .private.programming.primitives.primitive_tree import PrimitiveTree
from .private.programming.semantic import cx_semantic, mut_semantic
from .private.programming.window_ops import (
    add_window_ephemeral,
    add_window_primitives,
    ema,
)
from .private.programming.window_pair import (
    add_pair_window_primitives,
    rolling_beta,
    rolling_corr,
    rolling_cov,
)
from .private.programming.window_roll import (
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    rolling_sum,
)
from .private.programming.window_shift import delay, diff
from .private.programming.window_ts import add_ts_primitives, ts_argmax, ts_argmin, ts_rank

__all__ = [
    "Array",
    "Mask",
    "Window",
    "make_column_pset",
    "cx_one_point",
    "cx_one_point_leaf_biased",
    "generate",
    "gen_full",
    "gen_grow",
    "gen_half_and_half",
    "harm",
    "tree_to_infix",
    "mut_uniform",
    "mut_node_replacement",
    "mut_ephemeral",
    "mut_insert",
    "mut_shrink",
    "USER_DISPATCH_SIGNATURE",
    "numba_available",
    "bind_tape",
    "vadd",
    "vsub",
    "vmul",
    "vneg",
    "vabs",
    "vdiv",
    "vlog",
    "vsqrt",
    "vsin",
    "vcos",
    "vgt",
    "vlt",
    "vge",
    "vle",
    "veq",
    "vand",
    "vor",
    "vnot",
    "vwhere",
    "infer_fill",
    "add_numpy_primitives",
    "Opcode",
    "USER_BASE",
    "Tape",
    "BUILTIN_OPCODES",
    "bind_numba_opcode",
    "numba_opcodes",
    "lower_tree",
    "interpret_tape",
    "Terminal",
    "Ephemeral",
    "Primitive",
    "PrimitiveSet",
    "PrimitiveSetTyped",
    "PrimitiveTree",
    "cx_semantic",
    "mut_semantic",
    "compile_tree",
    "compile_adf_tree",
    "build_tree_graph",
    "static_limit",
    "ema",
    "add_window_primitives",
    "add_window_ephemeral",
    "add_pair_window_primitives",
    "add_ts_primitives",
    "delay",
    "diff",
    "rolling_sum",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "rolling_corr",
    "rolling_cov",
    "rolling_beta",
    "ts_rank",
    "ts_argmax",
    "ts_argmin",
]
