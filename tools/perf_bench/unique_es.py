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
"""deap-er-only CMA and restart benches."""

from __future__ import annotations

from deap_er import Toolbox
from deap_er import creator as er_creator
from deap_er import tools as er_tools

from .report import CaseResult
from .timing import mean_ms


def run_unique_es() -> list[CaseResult]:
    """Time boxed CMA, separable CMA, and IPOP/BIPOP restarts.

    Returns:
        Unique evolution-strategy cases.
    """
    ind_cls = er_creator.U_IND_SO

    def boxed_generate() -> None:
        er_tools.rng.seed(71)
        strategy = er_tools.Strategy(
            centroid=[0.0] * 8,
            sigma=1.0,
            offsprings=8,
            low=-2.0,
            up=2.0,
            bound_mode="clip",
        )
        strategy.generate(ind_cls)

    def separable() -> None:
        er_tools.rng.seed(72)
        strategy = er_tools.StrategySeparable(
            centroid=[0.0] * 8,
            sigma=1.0,
            offsprings=8,
            low=-2.0,
            up=2.0,
        )
        population = strategy.generate(ind_cls)
        for individual in population:
            individual.fitness.values = er_tools.bm_sphere(individual)
        strategy.update(population)

    def restarts() -> None:
        er_tools.rng.seed(73)
        strategy = er_tools.Strategy(centroid=[0.5] * 4, sigma=1.0, offsprings=6)
        restart = er_tools.RestartStrategy(strategy, mode="ipop", budget=24, sigma_large=1.0)
        toolbox = Toolbox()
        toolbox.register("evaluate", er_tools.bm_sphere)
        toolbox.register("generate", restart.generate, ind_cls)
        toolbox.register("update", restart.update)
        er_tools.ea_generate_update_restarts(toolbox, restart, log_restarts=False)

    return [
        CaseResult(
            name="Strategy boxed generate dim=8 λ=8",
            description="Box-constrained CMA generate (clip)",
            shared=False,
            deap_er_ms=mean_ms(boxed_generate, repeat=20, warmup=2),
            feature="boxed CMA",
            notes="repeat=20; dim=8, lambda=8",
            repeat=20,
            warmup=2,
        ),
        CaseResult(
            name="StrategySeparable generate+update dim=8",
            description="Separable CMA generate and update",
            shared=False,
            deap_er_ms=mean_ms(separable, repeat=20, warmup=2),
            feature="StrategySeparable",
            notes="repeat=20; one generate/update cycle",
            repeat=20,
            warmup=2,
        ),
        CaseResult(
            name="ea_generate_update_restarts budget=24",
            description="IPOP RestartStrategy loop",
            shared=False,
            deap_er_ms=mean_ms(restarts, repeat=10, warmup=1),
            feature="RestartStrategy",
            notes="repeat=10, warmup=1; dim=4, budget=24",
            repeat=10,
            warmup=1,
        ),
    ]
