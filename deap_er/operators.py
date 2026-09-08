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
from .private.operators.case_batch_reduce import (
    batch_case_matrix,
    partition_case_batches,
    reduce_case_mean,
    reduce_case_mse,
)
from .private.operators.case_exam_guard import guard_case_exams
from .private.operators.case_exam_step import next_lexicase_cases
from .private.operators.case_exams import score_case_exams
from .private.operators.constraint_dominates import constraint_dominates
from .private.operators.cx_hetero import cx_heterogeneous
from .private.operators.cx_permutation import (
    cx_ordered,
    cx_partially_matched,
    cx_uniform_partially_matched,
)
from .private.operators.cx_point import (
    cx_es_two_point,
    cx_es_two_point_copy,
    cx_messy_one_point,
    cx_one_point,
    cx_two_point,
    cx_two_point_copy,
)
from .private.operators.cx_real import (
    cx_blend,
    cx_blend_bounded,
    cx_es_blend,
    cx_simulated_binary,
    cx_simulated_binary_bounded,
    cx_uniform,
)
from .private.operators.downsample_schedule import next_downsample_cases
from .private.operators.island_eval_keys import island_eval_keys
from .private.operators.mig_ring import mig_fully_connected, mig_random, mig_ring
from .private.operators.mut_case_exam import mut_case_mask, mut_case_ranges
from .private.operators.mut_de import mut_de
from .private.operators.mut_gaussian_bounded import mut_gaussian_bounded
from .private.operators.mut_hetero import mut_heterogeneous
from .private.operators.mut_iso_line import iso_line_bit, iso_line_float, iso_line_int, mut_iso_line
from .private.operators.mut_various import (
    mut_es_log_normal,
    mut_flip_bit,
    mut_gaussian,
    mut_polynomial_bounded,
    mut_shuffle_indexes,
    mut_uniform_int,
)
from .private.operators.policy_action_guard import (
    PolicyActionGuard,
    estimate_policy_action_evals,
    guard_policy_action,
)
from .private.operators.policy_fitness import (
    guard_policy_fitness_exam,
    policy_held_out_fitness,
    resolve_policy_held_out,
)
from .private.operators.sample_informed_cases import sample_informed_cases
from .private.operators.sel_age_moea_2 import SelAGE2WithMemory, sel_age_moea_2
from .private.operators.sel_batch_epsilon_lexicase import sel_batch_epsilon_lexicase
from .private.operators.sel_helpers import assign_crowding_dist, uniform_reference_points
from .private.operators.sel_lexicase import sel_epsilon_lexicase, sel_lexicase
from .private.operators.sel_lexicase_matrix import fitness_case_matrix
from .private.operators.sel_moead import SelMOEADWithMemory, sel_moead
from .private.operators.sel_moead_helpers import (
    moead_neighborhood,
    scalarization_pbi,
    scalarization_tchebycheff,
)
from .private.operators.sel_novelty import sel_novelty
from .private.operators.sel_nsga_2 import sel_nsga_2
from .private.operators.sel_nsga_3 import SelNSGA3WithMemory, sel_nsga_3
from .private.operators.sel_sms_emoa import sel_sms_emoa
from .private.operators.sel_spea_2 import sel_spea_2
from .private.operators.sel_team import sel_team
from .private.operators.sel_tournament import sel_double_tournament, sel_tournament
from .private.operators.sel_tournament_cases import sel_tournament_cases
from .private.operators.sel_tournament_dcd import sel_tournament_dcd
from .private.operators.sel_various import (
    sel_best,
    sel_random,
    sel_roulette,
    sel_stochastic_universal_sampling,
    sel_worst,
)

__all__: list[str] = [
    "cx_ordered",
    "cx_partially_matched",
    "cx_uniform_partially_matched",
    "cx_es_two_point",
    "cx_es_two_point_copy",
    "cx_messy_one_point",
    "cx_one_point",
    "cx_two_point",
    "cx_two_point_copy",
    "cx_blend",
    "cx_blend_bounded",
    "cx_es_blend",
    "cx_simulated_binary",
    "cx_simulated_binary_bounded",
    "cx_uniform",
    "cx_heterogeneous",
    "island_eval_keys",
    "mig_fully_connected",
    "mig_random",
    "mig_ring",
    "mut_de",
    "mut_es_log_normal",
    "mut_flip_bit",
    "mut_gaussian",
    "mut_gaussian_bounded",
    "mut_heterogeneous",
    "iso_line_bit",
    "iso_line_float",
    "iso_line_int",
    "mut_iso_line",
    "mut_polynomial_bounded",
    "mut_shuffle_indexes",
    "mut_uniform_int",
    "assign_crowding_dist",
    "uniform_reference_points",
    "batch_case_matrix",
    "partition_case_batches",
    "reduce_case_mean",
    "reduce_case_mse",
    "fitness_case_matrix",
    "scalarization_tchebycheff",
    "scalarization_pbi",
    "moead_neighborhood",
    "sel_moead",
    "SelMOEADWithMemory",
    "sel_age_moea_2",
    "SelAGE2WithMemory",
    "sample_informed_cases",
    "score_case_exams",
    "guard_policy_fitness_exam",
    "policy_held_out_fitness",
    "resolve_policy_held_out",
    "guard_case_exams",
    "guard_policy_action",
    "PolicyActionGuard",
    "estimate_policy_action_evals",
    "next_lexicase_cases",
    "next_downsample_cases",
    "mut_case_mask",
    "mut_case_ranges",
    "sel_batch_epsilon_lexicase",
    "sel_epsilon_lexicase",
    "sel_lexicase",
    "sel_team",
    "sel_novelty",
    "constraint_dominates",
    "sel_nsga_2",
    "SelNSGA3WithMemory",
    "sel_nsga_3",
    "sel_spea_2",
    "sel_sms_emoa",
    "sel_double_tournament",
    "sel_tournament",
    "sel_tournament_cases",
    "sel_tournament_dcd",
    "sel_best",
    "sel_random",
    "sel_roulette",
    "sel_stochastic_universal_sampling",
    "sel_worst",
]
