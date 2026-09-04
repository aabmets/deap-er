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
from .crossover import (
    cx_blend,
    cx_es_blend,
    cx_es_two_point,
    cx_es_two_point_copy,
    cx_messy_one_point,
    cx_one_point,
    cx_ordered,
    cx_partially_matched,
    cx_simulated_binary,
    cx_simulated_binary_bounded,
    cx_two_point,
    cx_two_point_copy,
    cx_uniform,
    cx_uniform_partially_matched,
)
from .migration import mig_ring
from .mutation import (
    mut_es_log_normal,
    mut_flip_bit,
    mut_gaussian,
    mut_polynomial_bounded,
    mut_shuffle_indexes,
    mut_uniform_int,
)
from .selection import (
    SelNSGA3WithMemory,
    assign_crowding_dist,
    sel_best,
    sel_double_tournament,
    sel_epsilon_lexicase,
    sel_lexicase,
    sel_nsga_2,
    sel_nsga_3,
    sel_random,
    sel_roulette,
    sel_spea_2,
    sel_stochastic_universal_sampling,
    sel_tournament,
    sel_tournament_dcd,
    sel_worst,
    uniform_reference_points,
)

__all__ = [
    "SelNSGA3WithMemory",
    "assign_crowding_dist",
    "cx_blend",
    "cx_es_blend",
    "cx_es_two_point",
    "cx_es_two_point_copy",
    "cx_messy_one_point",
    "cx_one_point",
    "cx_ordered",
    "cx_partially_matched",
    "cx_simulated_binary",
    "cx_simulated_binary_bounded",
    "cx_two_point",
    "cx_two_point_copy",
    "cx_uniform",
    "cx_uniform_partially_matched",
    "mig_ring",
    "mut_es_log_normal",
    "mut_flip_bit",
    "mut_gaussian",
    "mut_polynomial_bounded",
    "mut_shuffle_indexes",
    "mut_uniform_int",
    "sel_best",
    "sel_double_tournament",
    "sel_epsilon_lexicase",
    "sel_lexicase",
    "sel_nsga_2",
    "sel_nsga_3",
    "sel_random",
    "sel_roulette",
    "sel_spea_2",
    "sel_stochastic_universal_sampling",
    "sel_tournament",
    "sel_tournament_dcd",
    "sel_worst",
    "uniform_reference_points",
]
