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
from .private.programming.affine_write import write_affine_scale
from .private.programming.columnar import Array, Mask, Window, make_column_pset
from .private.programming.columnar_setup import columnar_pset, evaluate_columnar
from .private.programming.compilers import (
    build_tree_graph,
    clear_compile_cache,
    compile_adf_tree,
    compile_tree,
    static_limit,
)
from .private.programming.crossover import cx_one_point, cx_one_point_leaf_biased
from .private.programming.generators import gen_full, gen_grow, gen_half_and_half, generate
from .private.programming.harm.harm import harm
from .private.programming.infix import tree_to_infix
from .private.programming.memetic import (
    assign_ephemerals,
    extract_ephemerals,
    numeric_leaves,
    tune_ephemerals,
)
from .private.programming.memetic_budget import tune_ephemerals_budget
from .private.programming.memetic_defaults import (
    MEMETIC_DEFAULT_N_GEN,
    MEMETIC_MAX_N_GEN,
    cap_tune_n_gen,
    estimate_tune_ephemerals_evals,
)
from .private.programming.mutation import (
    mut_ephemeral,
    mut_insert,
    mut_node_replacement,
    mut_shrink,
    mut_uniform,
)
from .private.programming.numba.numba_ops import (
    USER_DISPATCH_SIGNATURE,
    bind_tape,
    numba_available,
    warmup_numba,
)
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
from .private.programming.numpy.numpy_ops import add_numpy_primitives
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
from .private.programming.promote import promote_subtree, promoted_names
from .private.programming.register_gp import register_gp
from .private.programming.semantic import cx_semantic, mut_semantic
from .private.programming.slim.slim_ops import (
    cx_slim_donor,
    mut_slim,
    mut_slim_deflate,
    mut_slim_inflate,
)
from .private.programming.slim.slim_tree import SlimTree, compile_slim_tree
from .private.programming.suffix_rescore import suffix_rescore
from .private.programming.tape_batch import interpret_tapes
from .private.programming.tape_lookback import tape_lookback
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
from .private.various.affine_scale import affine_scale
from .private.various.semantic_descriptors import (
    semantic_descriptors,
    semantic_moments,
    semantic_solve_bits,
)
from .private.various.semantic_neighbors import semantic_distance, semantic_nearest
from .private.various.semantic_project import (
    semantic_pca_basis,
    semantic_project,
    semantic_random_basis,
)

__all__ = [
    "Array",
    "Mask",
    "Window",
    "make_column_pset",
    "columnar_pset",
    "evaluate_columnar",
    "register_gp",
    "cx_one_point",
    "cx_one_point_leaf_biased",
    "generate",
    "gen_full",
    "gen_grow",
    "gen_half_and_half",
    "harm",
    "tree_to_infix",
    "numeric_leaves",
    "extract_ephemerals",
    "assign_ephemerals",
    "tune_ephemerals",
    "tune_ephemerals_budget",
    "MEMETIC_DEFAULT_N_GEN",
    "MEMETIC_MAX_N_GEN",
    "cap_tune_n_gen",
    "estimate_tune_ephemerals_evals",
    "affine_scale",
    "write_affine_scale",
    "mut_uniform",
    "mut_node_replacement",
    "mut_ephemeral",
    "mut_insert",
    "mut_shrink",
    "USER_DISPATCH_SIGNATURE",
    "numba_available",
    "bind_tape",
    "warmup_numba",
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
    "add_numpy_primitives",
    "Opcode",
    "USER_BASE",
    "Tape",
    "BUILTIN_OPCODES",
    "bind_numba_opcode",
    "numba_opcodes",
    "lower_tree",
    "interpret_tape",
    "interpret_tapes",
    "tape_lookback",
    "suffix_rescore",
    "semantic_descriptors",
    "semantic_moments",
    "semantic_pca_basis",
    "semantic_project",
    "semantic_random_basis",
    "semantic_solve_bits",
    "semantic_distance",
    "semantic_nearest",
    "Terminal",
    "Ephemeral",
    "Primitive",
    "PrimitiveSet",
    "PrimitiveSetTyped",
    "PrimitiveTree",
    "cx_semantic",
    "mut_semantic",
    "SlimTree",
    "compile_slim_tree",
    "mut_slim_inflate",
    "mut_slim_deflate",
    "mut_slim",
    "cx_slim_donor",
    "compile_tree",
    "compile_adf_tree",
    "clear_compile_cache",
    "promote_subtree",
    "promoted_names",
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
