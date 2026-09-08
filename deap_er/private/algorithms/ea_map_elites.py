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
import time
from collections.abc import Callable, Sequence
from logging import Logger

from deap_er.private.records.archive_common import MapElitesArchive
from deap_er.private.records.logbook import Logbook
from deap_er.private.toolbox import Toolbox
from deap_er.private.typedefs import EvoStats, Individual

from .loop import budget_spent, check_n_evals, consume_evals, new_logbook
from .variation import var_or

__all__: list[str] = ["ea_map_elites"]


def _parent_pool(
    archive: MapElitesArchive,
    initial: list[Individual],
    batch_size: int,
    cx_prob: float,
) -> list[Individual]:
    """Build a parent list large enough for ``var_or`` crossover."""
    min_parents = 2 if cx_prob > 0 else 1
    pool_size = max(batch_size, min_parents)
    if len(archive):
        return archive.random_elites(pool_size, replace=True)
    if not initial:
        raise ValueError(
            "ea_map_elites requires a non-empty archive or initial population "
            "before variation generations"
        )
    return [initial[index % len(initial)] for index in range(pool_size)]


def _record_map_elites_generation(
    logbook,
    gen: int,
    nevals: int,
    archive: MapElitesArchive,
    population: list[Individual],
    stats: EvoStats | None,
    verbose: bool,
    logger: Logger | None,
    duration: float | None,
) -> None:
    record = stats.compile(population) if stats and population else {}
    archive_stats = archive.stats
    record["coverage"] = archive_stats.coverage
    record["num_elites"] = archive_stats.num_elites
    record["qd_score"] = archive_stats.qd_score
    if duration is not None:
        record["duration"] = duration
    logbook.record(gen=gen, nevals=nevals, **record)
    if verbose:
        text = logbook.stream
        if logger is not None:
            logger.info(text)
        else:
            print(text)


def ea_map_elites(
    toolbox: Toolbox,
    archive: MapElitesArchive,
    descriptor_fn: Callable[[Individual], Sequence[float]],
    initial: list[Individual],
    generations: int,
    batch_size: int,
    cx_prob: float,
    mut_prob: float,
    stats: EvoStats | None = None,
    verbose: bool = False,
    logger: Logger | None = None,
    log_time: bool = False,
    n_evals: int | None = None,
) -> tuple[MapElitesArchive, Logbook]:
    """Run MAP-Elites with ``var_or`` variation on archive elites.

    Generation zero evaluates ``initial`` and seeds the archive. Later
    generations sample parents from ``archive``, vary them with
    ``var_or``, evaluate the offspring, and try to improve cells.
    When ``n_evals`` is set, the generation that meets or exceeds that
    count is the last one recorded. Generations remain the default stop.

    Requires ``clone``, ``mate``, ``mutate``, and ``evaluate`` on
    ``toolbox``. ``archive`` stores single-objective fitness only.
    Accepts :class:`~deap_er.records.GridArchive`,
    :class:`~deap_er.records.CvtArchive`, or
    :class:`~deap_er.records.UnstructuredArchive`.

    Args:
        toolbox: Toolbox with the evolution operators.
        archive: MAP-Elites archive updated in place.
        descriptor_fn: Maps an evaluated individual to a behavior
            descriptor.
        initial: Individuals evaluated and archived before generation
            one.
        generations: Number of variation generations after the initial
            seeding generation.
        batch_size: Offspring produced each variation generation. When
            ``cx_prob`` is positive, the parent pool is at least two
            individuals so crossover can run.
        cx_prob: Probability of crossover in ``var_or``.
        mut_prob: Probability of mutation in ``var_or``.
        stats: Optional Statistics or MultiStatistics compiled from the
            offspring each generation. An empty seed or offspring list
            skips that compile so reducers such as ``max`` do not run
            on no data. Archive metrics still record.
        verbose: If True, print the logbook stream each generation.
        logger: If given with ``verbose``, the stream is logged.
        log_time: If True, record per-generation ``duration``.
        n_evals: Optional evaluation budget. The generation that
            meets or exceeds this count is finished, then the loop
            stops. ``None`` keeps the generation limit only.

    Returns:
        The archive and the logbook.

    Raises:
        ValueError: If a variation generation runs while the archive and
            ``initial`` are both empty, or if ``n_evals`` is negative.
    """
    check_n_evals(n_evals)
    logbook = new_logbook(stats, log_time=log_time)
    logbook.header = (
        ["gen", "nevals", "coverage", "num_elites", "qd_score"]
        + (["duration"] if log_time else [])
        + (stats.fields if stats else [])
    )

    t0 = time.perf_counter()
    nevals, used = consume_evals(toolbox, initial, n_evals, 0)
    for individual in initial:
        archive.add(individual, descriptor_fn(individual))
    duration = time.perf_counter() - t0 if log_time else None
    _record_map_elites_generation(
        logbook,
        0,
        nevals,
        archive,
        initial,
        stats,
        verbose,
        logger,
        duration,
    )
    if budget_spent(n_evals, used):
        return archive, logbook

    for gen in range(1, generations + 1):
        t0 = time.perf_counter()
        parents = _parent_pool(archive, initial, batch_size, cx_prob)
        offspring = var_or(toolbox, parents, batch_size, cx_prob, mut_prob)
        nevals, used = consume_evals(toolbox, offspring, n_evals, used)
        for individual in offspring:
            archive.add(individual, descriptor_fn(individual))
        duration = time.perf_counter() - t0 if log_time else None
        _record_map_elites_generation(
            logbook,
            gen,
            nevals,
            archive,
            offspring,
            stats,
            verbose,
            logger,
            duration,
        )
        if budget_spent(n_evals, used):
            break

    return archive, logbook
