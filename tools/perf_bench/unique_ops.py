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
"""deap-er-only operator and selection benches."""

from __future__ import annotations

from collections.abc import Callable

from deap_er import tools as er_tools

from .data import mo_data, so_data, wrap_mo, wrap_so
from .report import CaseResult
from .timing import REPEAT, WARMUP, mean_ms


def _case(
    name: str,
    description: str,
    feature: str,
    func: Callable[[], object],
    *,
    notes: str = "",
    repeat: int = REPEAT,
    warmup: int = WARMUP,
) -> CaseResult:
    """Time one unique case.

    Args:
        name: Case label.
        description: Short description.
        feature: Inventory capability name.
        func: Nullary callable to time.
        notes: Optional workload note.
        repeat: Timed repetitions.
        warmup: Untimed calls first.

    Returns:
        A unique ``CaseResult``.
    """
    return CaseResult(
        name=name,
        description=description,
        shared=False,
        deap_er_ms=mean_ms(func, repeat=repeat, warmup=warmup),
        feature=feature,
        notes=notes,
        repeat=repeat,
        warmup=warmup,
    )


def run_unique_ops() -> list[CaseResult]:
    """Time deap-er-only operators and selectors.

    Returns:
        Unique operator cases.
    """
    genes, fits = mo_data(40, 8, 3, 51)
    pop = wrap_mo(genes, fits)
    pop_dcd = wrap_mo(*mo_data(32, 8, 3, 52))
    er_tools.assign_crowding_dist(pop_dcd)
    real = wrap_mo(*mo_data(40, 10, 3, 53))
    binary = wrap_so(so_data(40, 16, 54))
    cases_pop = wrap_mo(*mo_data(60, 8, 80, 55), many=True)
    weights = er_tools.uniform_reference_points(3, ref_ppo=4)
    donors = [list(ind) for ind in real[:4]]

    def blend() -> None:
        er_tools.rng.seed(61)
        left, right = list(real[0]), list(real[1])
        er_tools.cx_blend_bounded(left, right, 0.5, 0.0, 1.0)

    def gauss_box() -> None:
        er_tools.rng.seed(62)
        er_tools.mut_gaussian_bounded(list(real[2]), 0.0, 0.2, 0.0, 1.0, 0.3)

    def hetero() -> None:
        er_tools.rng.seed(63)
        mutators = [lambda gene: 1 - gene for _ in binary[0]]
        er_tools.mut_heterogeneous(list(binary[0]), mutators, 0.4)
        er_tools.cx_heterogeneous(
            list(binary[1]), list(binary[2]), [lambda a, b: (b, a)] * len(binary[1])
        )

    def de_trial() -> None:
        er_tools.rng.seed(64)
        trial = list(donors[0])
        er_tools.mut_de(trial, donors[1], donors[2], donors[3], 0.5, 0.7, low=0.0, up=1.0)

    def crowd_w() -> None:
        er_tools.assign_crowding_dist(pop, use_weights=True)

    def tourney_dcd() -> None:
        er_tools.rng.seed(65)
        er_tools.sel_tournament_dcd(pop_dcd, 10)

    def informed() -> None:
        er_tools.rng.seed(66)
        matrix = er_tools.fitness_case_matrix(cases_pop)
        subset = er_tools.sample_informed_cases(cases_pop, 16, matrix=matrix, trust_matrix=True)
        er_tools.sel_lexicase(cases_pop, 20, cases=subset, matrix=matrix, trust_matrix=True)
        er_tools.sel_epsilon_lexicase(
            cases_pop, 20, 0.05, cases=subset, matrix=matrix, trust_matrix=True
        )

    def team() -> None:
        er_tools.rng.seed(67)
        er_tools.sel_team(cases_pop, 8)

    def exams() -> None:
        er_tools.rng.seed(68)
        elites = cases_pop[:12]
        pool = [er_tools.CaseExam.from_cases(list(range(i, i + 8)), 80) for i in range(0, 32, 8)]
        er_tools.score_case_exams(pool, elites)
        er_tools.next_lexicase_cases(pool, elites, case_count=8)

    def sms() -> None:
        er_tools.sel_sms_emoa(pop, 20)

    def moead() -> None:
        er_tools.sel_moead(pop, 20, weights)

    def age() -> None:
        er_tools.sel_age_moea_2(pop, 16)

    def constrained() -> None:
        def feasible(ind: object) -> bool:
            return ind[0] < 0.7

        def violation(ind: object) -> float:
            return max(0.0, ind[0] - 0.7)

        for left in pop[:12]:
            for right in pop[12:24]:
                er_tools.constraint_dominates(left, right, feasible=feasible, violation=violation)
        er_tools.sel_nsga_2(pop, 20, feasible=feasible, violation=violation)

    return [
        _case("cx_blend_bounded n=10", "Boxed blend crossover", "cx_blend_bounded", blend),
        _case(
            "mut_gaussian_bounded n=10",
            "Boxed Gaussian mutation",
            "mut_gaussian_bounded",
            gauss_box,
        ),
        _case(
            "mut/cx_heterogeneous n=16", "Per-gene mutator and mate", "mut_heterogeneous", hetero
        ),
        _case("mut_de n=10", "DE/rand/1/bin trial", "mut_de", de_trial),
        _case(
            "assign_crowding_dist use_weights n=40",
            "Crowding on wvalues",
            "assign_crowding_dist use_weights",
            crowd_w,
        ),
        _case(
            "sel_tournament_dcd n=32 k=10",
            "DCD tournament for any valid k",
            "sel_tournament_dcd",
            tourney_dcd,
        ),
        _case(
            "informed lexicase n=60 cases=80",
            "fitness_case_matrix + sample_informed_cases + cases=",
            "sel_lexicase cases=",
            informed,
        ),
        _case("sel_team n=60 k=8", "Greedy max-coverage team", "sel_team", team),
        _case(
            "score_case_exams + next_lexicase_cases",
            "Co-evolving case exams",
            "CaseExam",
            exams,
        ),
        _case("sel_sms_emoa n=40 k=20", "SMS-EMOA hypervolume truncate", "sel_sms_emoa", sms),
        _case("sel_moead n=40 k=20", "MOEA/D Tchebycheff selection", "sel_moead", moead),
        _case(
            "sel_age_moea_2 n=40 k=16",
            "AGE-MOEA-II geodesic selection",
            "sel_age_moea_2",
            age,
            notes="smaller pool than shared NSGA cases",
        ),
        _case(
            "sel_nsga_2 constrained n=40 k=20",
            "constraint_dominates + feasible=/violation=",
            "constraint_dominates",
            constrained,
        ),
    ]
