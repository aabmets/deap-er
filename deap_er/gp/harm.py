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
from typing import Any

from deap_er.base import Toolbox
from deap_er.private.algorithms.loop import evaluate_invalid
from deap_er.records import Logbook
from deap_er.records.typedefs import AlgoResult, Hof, Stats

from ._harm_breed import _produce
from ._harm_size import (
    _acceptance,
    _cutoff_size,
    _natural_histogram,
    _target_histogram,
    _target_prob,
)
from .typedefs import GPIndividual

__all__ = ["harm"]

_HARM_DEFAULTS: dict[str, float | int] = {
    "alpha": 0.05,
    "beta": 10.0,
    "gamma": 0.25,
    "rho": 0.9,
    "nb_model": -1,
    "min_cutoff": 20,
}


def harm(
    toolbox: Toolbox,
    population: list[GPIndividual],
    generations: int,
    cx_prob: float,
    mut_prob: float,
    hof: Hof | None = None,
    stats: Stats | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> AlgoResult:
    """Evolve a GP population with HARM bloat control.

    Default parameter values are recommended for most use cases.
    Requires ``mate``, ``mutate``, ``select``, ``evaluate``, ``clone``,
    and ``map`` on ``toolbox``.

    Args:
        toolbox: Toolbox with the evolution operators.
        population: Individuals to evolve. Replaced in place.
        generations: Number of generations to run.
        cx_prob: Probability of mating two individuals.
        mut_prob: Probability of mutating an individual.
        hof: Optional HallOfFame or ParetoFront to update.
        stats: Optional Statistics or MultiStatistics to compile.
        verbose: If True, print the logbook stream each generation.
        **kwargs: HARM size-control knobs. Keyword-only. Accepted
            keys:

            * ``alpha``: Half-life of the exponential, scaled
              linearly with the cutoff. Higher values accept larger
              individuals. Default ``0.05``.
            * ``beta``: Minimum half-life, so growth remains
              possible while individuals are still small. Default
              ``10.0``.
            * ``gamma``: Fraction of individuals allowed past the
              cutoff. Default ``0.25``.
            * ``rho``: Fitness range used to place the cutoff.
              Higher values search more aggressively for slightly
              better solutions and may overfit. Default ``0.9``.
            * ``nb_model``: Individuals generated to model the
              natural size distribution. ``-1`` uses
              ``max(2000, len(population))``. Default ``-1``.
            * ``min_cutoff``: Absolute minimum cutoff, to avoid
              shrinking the population too early. Default ``20``.

    Returns:
        The final population and the logbook.

    Raises:
        TypeError: If ``kwargs`` contains an unknown name.
    """
    unknown = kwargs.keys() - _HARM_DEFAULTS.keys()
    if unknown:
        names = ", ".join(repr(name) for name in sorted(unknown))
        raise TypeError(f"harm() got unexpected keyword argument(s): {names}")

    alpha = float(kwargs.get("alpha", _HARM_DEFAULTS["alpha"]))
    beta = float(kwargs.get("beta", _HARM_DEFAULTS["beta"]))
    gamma = float(kwargs.get("gamma", _HARM_DEFAULTS["gamma"]))
    rho = float(kwargs.get("rho", _HARM_DEFAULTS["rho"]))
    nb_model = int(kwargs.get("nb_model", _HARM_DEFAULTS["nb_model"]))
    min_cutoff = int(kwargs.get("min_cutoff", _HARM_DEFAULTS["min_cutoff"]))

    logbook = Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    nevals = evaluate_invalid(toolbox, population)

    if hof is not None:
        hof.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=nevals, **record)

    if verbose:
        print(logbook.stream)

    if nb_model == -1:
        nb_model = max(2000, len(population))

    for gen in range(1, generations + 1):
        pop_len = len(population)
        natural_pop, natural_pop_sizes = _produce(toolbox, population, nb_model, cx_prob, mut_prob)
        natural_hist = _natural_histogram(natural_pop_sizes, pop_len, nb_model)
        cutoff_size = _cutoff_size(population, pop_len, rho, min_cutoff)

        def target_prob(size: int, cutoff: int = cutoff_size, length: int = pop_len) -> float:
            return _target_prob(size, alpha, beta, gamma, length, cutoff)

        target_hist = _target_histogram(natural_hist, cutoff_size, target_prob)
        accept_func = _acceptance(natural_hist, target_hist, target_prob)

        offspring, _ = _produce(
            toolbox, population, pop_len, cx_prob, mut_prob, natural_pop, accept_func
        )

        nevals = evaluate_invalid(toolbox, offspring)

        if hof is not None:
            hof.update(offspring)

        population[:] = offspring
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=nevals, **record)

        if verbose:
            print(logbook.stream)

    return population, logbook
