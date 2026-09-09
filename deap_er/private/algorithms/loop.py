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
from collections.abc import Sequence
from logging import Logger
from typing import Any

from deap_er.private.toolbox import Toolbox
from deap_er.private.typedefs import EvoRecords, EvoStats, Individual
from deap_er.records import Logbook, ParetoFront

__all__: list[str] = [
    "budget_spent",
    "check_n_evals",
    "consume_evals",
    "evaluate_invalid",
    "new_logbook",
    "record_generation",
]


def new_logbook(stats: EvoStats | None, log_time: bool = False) -> Logbook:
    """Create a logbook with the standard algorithm header.

    Args:
        stats: Optional Statistics or MultiStatistics whose fields
            become the trailing header columns.
        log_time: If True, include a ``duration`` column.

    Returns:
        A logbook ready to record generations.
    """
    logbook = Logbook()
    extra = ["duration"] if log_time else []
    logbook.header = ["gen", "nevals"] + extra + (stats.fields if stats else [])
    return logbook


def check_n_evals(n_evals: int | None) -> None:
    """Reject a negative evaluation budget.

    Args:
        n_evals: Optional maximum number of evaluations.

    Raises:
        ValueError: If ``n_evals`` is negative.
    """
    if n_evals is not None and n_evals < 0:
        raise ValueError("n_evals must be at least 0.")


def budget_spent(n_evals: int | None, used: int) -> bool:
    """Return whether an optional evaluation budget is exhausted.

    Args:
        n_evals: Optional maximum number of evaluations.
        used: Evaluations already consumed.

    Returns:
        True when ``n_evals`` is set and ``used`` has reached it.
    """
    return n_evals is not None and used >= n_evals


def consume_evals(
    toolbox: Toolbox,
    individuals: Sequence[Any],
    n_evals: int | None,
    used: int,
) -> tuple[int, int]:
    """Evaluate invalids unless the evaluation budget is already spent.

    Args:
        toolbox: Toolbox with the evaluate and map operators.
        individuals: Individuals to scan for invalid fitness.
        n_evals: Optional maximum number of evaluations.
        used: Evaluations already consumed.

    Returns:
        The number evaluated this call and the new total used.
    """
    if budget_spent(n_evals, used):
        return 0, used
    nevals = evaluate_invalid(toolbox, individuals)
    return nevals, used + nevals


def evaluate_invalid(toolbox: Toolbox, individuals: Sequence[Any]) -> int:
    """Evaluate the individuals whose fitness is invalid.

    This is the helper ``ea_*`` drivers and ``apply_policy_action``
    already use. When the toolbox has an ``evaluate_batch`` operator,
    the whole batch of invalid individuals is handed to it in one
    call and ``map`` is not used. Otherwise each individual goes
    through ``map`` and ``evaluate`` as usual.

    Args:
        toolbox: Toolbox with the evaluate and map operators.
        individuals: Individuals to scan for invalid fitness.

    Returns:
        The number of individuals that were evaluated.
    """
    invalids = [ind for ind in individuals if not ind.fitness.is_valid()]
    evaluate_batch = getattr(toolbox, "evaluate_batch", None)
    if evaluate_batch is not None:
        fitness = evaluate_batch(invalids)
    else:
        fitness = toolbox.map(toolbox.evaluate, invalids)
    for ind, fit in zip(invalids, fitness, strict=False):
        ind.fitness.values = fit
    return len(invalids)


def record_generation(
    logbook: Logbook,
    gen: int,
    nevals: int,
    *,
    population: list[Individual],
    offspring: list[Individual],
    hof: EvoRecords | None,
    stats: EvoStats | None,
    verbose: bool,
    logger: Logger | None = None,
    duration: float | None = None,
    fronts: list[Any] | None = None,
) -> None:
    """Update the hall of fame and append one generation to the logbook.

    The hall of fame sees ``offspring``, while statistics are compiled
    from ``population``. The two differ whenever survivors are chosen
    from a pool larger than the offspring.

    Args:
        logbook: Logbook to append the record to.
        gen: Generation number of the record.
        nevals: Number of individuals evaluated this generation.
        population: Individuals to compile statistics from.
        offspring: Individuals to offer to the hall of fame.
        hof: Optional HallOfFame or ParetoFront to update.
        stats: Optional Statistics or MultiStatistics to compile.
            Skipped when ``population`` is empty so reducers such as
            ``max`` do not run on no data.
        verbose: If True, emit the logbook stream.
        logger: If given, the stream is logged instead of printed.
        duration: Optional wall time of this generation in seconds.
        fronts: Optional list that receives a ParetoFront snapshot
            of ``population`` for this generation.
    """
    if hof is not None:
        hof.update(offspring)
    if fronts is not None:
        front = ParetoFront()
        front.update(population)
        fronts.append(front)
    record = stats.compile(population) if stats and population else {}
    if duration is not None:
        record["duration"] = duration
    logbook.record(gen=gen, nevals=nevals, **record)
    if verbose:
        text = logbook.stream
        if logger is not None:
            logger.info(text)
        else:
            print(text)
