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
"""deap-er-only archive, island, and records benches."""

from __future__ import annotations

import numpy
from deap_er import Toolbox
from deap_er import tools as er_tools

from .data import so_data, wrap_so
from .report import CaseResult
from .timing import mean_ms


def _behavior(individual: list) -> tuple[float]:
    """Map a binary individual to a one-dimensional descriptor.

    Args:
        individual: Bit-string individual.

    Returns:
        Descriptor used by MAP-Elites archives.
    """
    return (float(sum(individual)),)


def run_unique_qd() -> list[CaseResult]:
    """Time MAP-Elites archives, islands, and records extras.

    Returns:
        Unique records and QD cases.
    """
    pop = wrap_so(so_data(32, 10, 91), map_elites=True)
    binaries = wrap_so(so_data(24, 16, 92))
    centroids = numpy.linspace(0.0, 10.0, 12).reshape(-1, 1)

    def grid_add() -> None:
        archive = er_tools.GridArchive(ranges=[(0.0, 10.0)], bins=8)
        for individual in pop:
            archive.add(individual, _behavior(individual))
        archive.random_elites(4)
        _ = archive.stats.coverage

    def cvt_unstruct() -> None:
        cvt = er_tools.CvtArchive(centroids)
        unstructured = er_tools.UnstructuredArchive(1, min_distance=1.0, max_elites=16)
        for individual in pop:
            desc = _behavior(individual)
            cvt.add(individual, desc)
            unstructured.add(individual, desc)

    def map_elites() -> None:
        toolbox = Toolbox()
        toolbox.register("mate", er_tools.cx_two_point)
        toolbox.register("mutate", er_tools.mut_flip_bit, mut_prob=0.2)
        toolbox.register("evaluate", lambda individual: (float(sum(individual)),))
        archive = er_tools.GridArchive(ranges=[(0.0, 16.0)], bins=8)
        er_tools.rng.seed(93)
        er_tools.ea_map_elites(
            toolbox,
            archive,
            _behavior,
            wrap_so(so_data(16, 8, 93), map_elites=True),
            generations=3,
            batch_size=12,
            cx_prob=0.5,
            mut_prob=0.2,
        )

    def islands() -> None:
        def vary(population: list) -> list:
            return list(population) + wrap_so(so_data(4, 10, 94), map_elites=True)

        toolbox = Toolbox()
        toolbox.register("evaluate", lambda individual: (float(sum(individual)),))
        toolbox.register("vary", vary)
        toolbox.register("select", er_tools.sel_best)
        left = wrap_so(so_data(12, 10, 95), map_elites=True)
        right = wrap_so(so_data(12, 10, 96), map_elites=True)

        def migrate(populations: list) -> None:
            er_tools.mig_ring(populations, 2, er_tools.sel_random)

        er_tools.step_islands([(toolbox, left), (toolbox, right)], migrate=migrate)

    def chapters() -> None:
        stats = er_tools.MultiStatistics(
            fitness=er_tools.Statistics(lambda ind: ind.fitness.values[0]),
            size=er_tools.Statistics(len),
        )
        stats.register("avg", numpy.mean, chapters=["fitness"])
        stats.register("max", max, chapters="size")
        stats.compile(binaries)

    def duplicates() -> None:
        er_tools.duplicate_count(binaries)

    def ea_extras() -> None:
        toolbox = Toolbox()
        toolbox.register("mate", er_tools.cx_two_point)
        toolbox.register("mutate", er_tools.mut_flip_bit, mut_prob=0.2)
        toolbox.register("select", er_tools.sel_tournament, contestants=3)
        toolbox.register("evaluate", lambda individual: (float(sum(individual)),))
        fronts: list = []
        er_tools.rng.seed(97)
        er_tools.ea_simple(
            toolbox,
            wrap_so(so_data(16, 12, 97)),
            4,
            0.5,
            0.2,
            log_time=True,
            fronts=fronts,
        )

    return [
        CaseResult(
            name="GridArchive.add n=32",
            description="MAP-Elites grid insert and elite sample",
            shared=False,
            deap_er_ms=mean_ms(grid_add),
            feature="GridArchive",
        ),
        CaseResult(
            name="CvtArchive + UnstructuredArchive n=32",
            description="CVT and unstructured archive inserts",
            shared=False,
            deap_er_ms=mean_ms(cvt_unstruct),
            feature="CvtArchive",
        ),
        CaseResult(
            name="ea_map_elites gens=3 n=16",
            description="MAP-Elites evaluate → archive → var_or",
            shared=False,
            deap_er_ms=mean_ms(map_elites, repeat=10, warmup=1),
            feature="ea_map_elites",
            notes="repeat=10, warmup=1; 3 generations",
            repeat=10,
            warmup=1,
        ),
        CaseResult(
            name="step_islands 2 demes n=12",
            description="Heterogeneous island step + mig_ring",
            shared=False,
            deap_er_ms=mean_ms(islands, repeat=20, warmup=2),
            feature="step_islands",
            notes="repeat=20",
            repeat=20,
            warmup=2,
        ),
        CaseResult(
            name="MultiStatistics chapters= n=24",
            description="Chapter-targeted statistic register",
            shared=False,
            deap_er_ms=mean_ms(chapters),
            feature="MultiStatistics chapters=",
        ),
        CaseResult(
            name="duplicate_count n=24",
            description="Variety statistic",
            shared=False,
            deap_er_ms=mean_ms(duplicates),
            feature="duplicate_count",
        ),
        CaseResult(
            name="ea_simple log_time+fronts n=16 gens=4",
            description="ea_* log_time and per-generation fronts",
            shared=False,
            deap_er_ms=mean_ms(ea_extras, repeat=20, warmup=2),
            feature="ea_* log_time/fronts",
            notes="repeat=20; logger omitted (I/O hook)",
            repeat=20,
            warmup=2,
        ),
    ]
