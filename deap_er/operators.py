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
from .private.operators.cx_hetero import cx_heterogeneous
from .private.operators.cx_real import (
    cx_blend,
    cx_blend_bounded,
    cx_es_blend,
    cx_simulated_binary,
    cx_simulated_binary_bounded,
    cx_uniform,
)
from .private.operators.mig_ring import mig_ring
from .private.operators.mut_gaussian_bounded import mut_gaussian_bounded
from .private.operators.mut_hetero import mut_heterogeneous
from .private.operators.mut_various import (
    mut_es_log_normal,
    mut_flip_bit,
    mut_gaussian,
    mut_polynomial_bounded,
    mut_shuffle_indexes,
    mut_uniform_int,
)
from .private.operators.sample_informed_cases import sample_informed_cases
from .private.operators.sel_age_moea_2 import SelAGE2WithMemory, sel_age_moea_2
from .private.operators.sel_helpers import assign_crowding_dist, uniform_reference_points
from .private.operators.sel_lexicase import sel_epsilon_lexicase, sel_lexicase
from .private.operators.sel_lexicase_matrix import fitness_case_matrix
from .private.operators.sel_moead import SelMOEADWithMemory, sel_moead
from .private.operators.sel_moead_helpers import (
    moead_neighborhood,
    scalarization_pbi,
    scalarization_tchebycheff,
)
from .private.operators.sel_nsga_2 import sel_nsga_2
from .private.operators.sel_nsga_3 import SelNSGA3WithMemory, sel_nsga_3
from .private.operators.sel_sms_emoa import sel_sms_emoa
from .private.operators.sel_spea_2 import sel_spea_2
from .private.operators.sel_tournament import sel_double_tournament, sel_tournament
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
    "mig_ring",
    "mut_es_log_normal",
    "mut_flip_bit",
    "mut_gaussian",
    "mut_gaussian_bounded",
    "mut_heterogeneous",
    "mut_polynomial_bounded",
    "mut_shuffle_indexes",
    "mut_uniform_int",
    "assign_crowding_dist",
    "uniform_reference_points",
    "fitness_case_matrix",
    "scalarization_tchebycheff",
    "scalarization_pbi",
    "moead_neighborhood",
    "sel_moead",
    "SelMOEADWithMemory",
    "sel_age_moea_2",
    "SelAGE2WithMemory",
    "sample_informed_cases",
    "sel_epsilon_lexicase",
    "sel_lexicase",
    "sel_nsga_2",
    "SelNSGA3WithMemory",
    "sel_nsga_3",
    "sel_spea_2",
    "sel_sms_emoa",
    "sel_double_tournament",
    "sel_tournament",
    "sel_tournament_dcd",
    "sel_best",
    "sel_random",
    "sel_roulette",
    "sel_stochastic_universal_sampling",
    "sel_worst",
]
