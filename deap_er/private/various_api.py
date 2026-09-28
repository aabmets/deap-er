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
from .various.affine_case_errors import affine_case_errors
from .various.affine_scale import affine_scale
from .various.bin2float import bin2float
from .various.case_errors import case_errors, case_intervals, case_valid_mask
from .various.case_generalization import (
    CaseGeneralizationRecipe,
    case_generalization_pool,
    case_generalization_recipe,
    held_out_tail,
    make_lexicase_train_select,
    train_head,
)
from .various.case_halving import (
    CaseHalvingResult,
    case_eval_charge,
    case_halving_stages,
    evaluate_case_halving,
    subset_evaluate_cases,
)
from .various.clone import clone_individual
from .various.constraints import ClosestValidPenalty, DeltaPenalty
from .various.decorators import Noise, Rotation, Scaling, Translation
from .various.eval_cache import EvalCache, clear_eval_caches, invalidate_eval
from .various.fitness_race import RaceStopResult, race_eval_charge, race_stop
from .various.fitness_resample import noisy_draw_key, resample, resample_aggregate
from .various.hypervolume import hypervolume
from .various.initializers import init_cycle, init_iterate, init_repeat
from .various.least_contrib import least_contrib
from .various.metrics import duplicate_count, inv_gen_dist, nsga_convergence, nsga_diversity
from .various.policy_observe import (
    policy_exam_scores,
    policy_observe,
    policy_promoted_library_size,
    policy_solve_bits_from_errors,
    policy_solve_bits_from_fitness,
    policy_solve_bits_from_semantic_row,
    policy_unsolved_count,
)
from .various.rng import RNG, rng
from .various.rng_spawn import bind_spawned_rng, call_spawned, map_spawned, spawn_rng
from .various.semantic_descriptors import (
    semantic_descriptors,
    semantic_moments,
    semantic_solve_bits,
)
from .various.semantic_mask import semantic_valid_mask
from .various.semantic_neighbors import semantic_distance, semantic_nearest
from .various.semantic_project import semantic_pca_basis, semantic_project, semantic_random_basis
from .various.sort_non_dominated import sort_non_dominated
from .various.sorting_network import SortingNetwork
from .various.structural_meta_case import (
    STRUCTURAL_META_CASES,
    structural_meta_case_columns,
    structural_meta_case_weights,
)

__all__ = [
    "affine_case_errors",
    "affine_scale",
    "bin2float",
    "case_errors",
    "case_intervals",
    "case_valid_mask",
    "CaseGeneralizationRecipe",
    "case_generalization_pool",
    "case_generalization_recipe",
    "held_out_tail",
    "make_lexicase_train_select",
    "train_head",
    "CaseHalvingResult",
    "case_eval_charge",
    "case_halving_stages",
    "evaluate_case_halving",
    "subset_evaluate_cases",
    "clone_individual",
    "DeltaPenalty",
    "ClosestValidPenalty",
    "Translation",
    "Rotation",
    "Scaling",
    "Noise",
    "EvalCache",
    "clear_eval_caches",
    "invalidate_eval",
    "RaceStopResult",
    "race_eval_charge",
    "race_stop",
    "noisy_draw_key",
    "resample",
    "resample_aggregate",
    "hypervolume",
    "init_repeat",
    "init_iterate",
    "init_cycle",
    "least_contrib",
    "nsga_diversity",
    "nsga_convergence",
    "inv_gen_dist",
    "duplicate_count",
    "policy_exam_scores",
    "policy_observe",
    "policy_promoted_library_size",
    "policy_solve_bits_from_errors",
    "policy_solve_bits_from_fitness",
    "policy_solve_bits_from_semantic_row",
    "policy_unsolved_count",
    "RNG",
    "rng",
    "bind_spawned_rng",
    "call_spawned",
    "map_spawned",
    "spawn_rng",
    "semantic_descriptors",
    "semantic_moments",
    "semantic_solve_bits",
    "semantic_valid_mask",
    "semantic_distance",
    "semantic_nearest",
    "semantic_pca_basis",
    "semantic_project",
    "semantic_random_basis",
    "sort_non_dominated",
    "SortingNetwork",
    "STRUCTURAL_META_CASES",
    "structural_meta_case_columns",
    "structural_meta_case_weights",
]
