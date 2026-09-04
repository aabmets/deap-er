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
import math
from collections.abc import Callable
from typing import Any

from deap_er.base import Toolbox
from deap_er.records import Logbook
from deap_er.records.dtypes import AlgoResult, Hof, Stats
from deap_er.rng import rng

from .dtypes import GPIndividual

__all__ = ["harm"]

_HARM_DEFAULTS: dict[str, float | int] = {
    "alpha": 0.05,
    "beta": 10.0,
    "gamma": 0.25,
    "rho": 0.9,
    "nb_model": -1,
    "min_cutoff": 20,
}


def _accept_all(_size: int) -> bool:
    """Accept an individual of any size.

    Args:
        _size: Size of the candidate individual, ignored.

    Returns:
        Always True.
    """
    return True


def _target_prob(
    size: int, alpha: float, beta: float, gamma: float, pop_len: int, cutoff_size: int
) -> float:
    """Return the target acceptance probability for a given tree size.

    The probability decays exponentially past ``cutoff_size``, with a
    half-life that grows linearly with the size.

    Args:
        size: Tree size to score.
        alpha: Half-life scaling factor.
        beta: Minimum half-life.
        gamma: Fraction of individuals allowed past the cutoff.
        pop_len: Number of individuals in the population.
        cutoff_size: Size at which the decay starts.

    Returns:
        The target probability for that size.
    """
    half_life = size * float(alpha) + beta
    hl_1 = gamma * pop_len * math.log(2) / half_life
    hl_2 = math.exp(-math.log(2) * (size - cutoff_size) / half_life)
    return hl_1 * hl_2


def _natural_histogram(sizes: list[int], pop_len: int, nb_model: int) -> list[float]:
    """Build the smoothed size distribution of the natural population.

    Each individual contributes to its own size bin and, with smaller
    weights, to the neighbouring bins.

    Args:
        sizes: Sizes of the modelled individuals.
        pop_len: Number of individuals in the population.
        nb_model: Number of individuals used to build the model.

    Returns:
        The histogram, scaled to the population size.
    """
    hist: list[float] = [0.0] * (max(sizes) + 3)
    for ind_size in sizes:
        hist[ind_size] += 0.4
        hist[ind_size - 1] += 0.2
        hist[ind_size + 1] += 0.2
        hist[ind_size + 2] += 0.1
        if ind_size - 2 >= 0:
            hist[ind_size - 2] += 0.1
    return [val * pop_len / nb_model for val in hist]


def _cutoff_size(natural_pop: list[GPIndividual], pop_len: int, rho: float, min_cutoff: int) -> int:
    """Return the tree size at which the size penalty starts.

    Args:
        natural_pop: Individuals modelling the natural distribution.
        pop_len: Number of individuals in the population.
        rho: Fitness range used to place the cutoff.
        min_cutoff: Absolute minimum cutoff.

    Returns:
        The cutoff size.
    """
    sorted_natural = sorted(natural_pop, key=lambda ind: ind.fitness)
    cutoff_candidates = sorted_natural[int(pop_len * rho - 1) :]
    return max(min_cutoff, len(min(cutoff_candidates, key=len)))


def _target_histogram(
    natural_hist: list[float], cutoff_size: int, target_prob: Callable[[int], float]
) -> list[float]:
    """Build the desired size distribution of the next generation.

    Bins up to ``cutoff_size`` keep their natural frequency; larger
    bins are replaced by the decaying target.

    Args:
        natural_hist: Natural size distribution.
        cutoff_size: Size at which the decay starts.
        target_prob: Callable returning the target for one size.

    Returns:
        The target histogram, aligned with ``natural_hist``.
    """
    target_hist = []
    for bin_idx in range(len(natural_hist)):
        if bin_idx <= cutoff_size:
            target_hist.append(natural_hist[bin_idx])
        else:
            target_hist.append(target_prob(bin_idx))
    return target_hist


def _acceptance(
    natural_hist: list[float], target_hist: list[float], target_prob: Callable[[int], float]
) -> Callable[[int], bool]:
    """Build the size-based acceptance test for one generation.

    Args:
        natural_hist: Natural size distribution.
        target_hist: Desired size distribution.
        target_prob: Callable returning the target for one size.

    Returns:
        A callable that randomly accepts an individual of a given size.
    """
    prob_hist = [t / n if n > 0 else t for n, t in zip(natural_hist, target_hist, strict=False)]

    def accept(size: int) -> bool:
        prob = prob_hist[size] if size < len(prob_hist) else target_prob(size)
        return rng.random() <= prob

    return accept


def _produce(
    toolbox: Toolbox,
    population: list[GPIndividual],
    count: int,
    cx_prob: float,
    mut_prob: float,
    pick_from: list[GPIndividual] | None = None,
    accept_func: Callable[[int], bool] = _accept_all,
) -> tuple[list[GPIndividual], list[int]]:
    """Produce individuals until ``count`` of them pass ``accept_func``.

    Candidates are taken from ``pick_from`` while it lasts, and are
    otherwise bred from ``population`` by crossover, mutation, or
    reproduction.

    Args:
        toolbox: Toolbox with the variation operators.
        population: Individuals to breed from.
        count: Number of individuals to produce.
        cx_prob: Probability of producing a child by crossover.
        mut_prob: Probability of producing a child by mutation.
        pick_from: Optional pool of ready-made candidates, consumed
            from the end.
        accept_func: Predicate on candidate size.

    Returns:
        The produced individuals and their sizes.
    """
    if pick_from is None:
        pick_from = []

    produced_pop: list[Any] = []
    produced_pop_sizes: list[int] = []

    while len(produced_pop) < count:
        if len(pick_from) > 0:
            aspirant = pick_from.pop()
            if accept_func(len(aspirant)):
                produced_pop.append(aspirant)
                produced_pop_sizes.append(len(aspirant))
        else:
            op_random = rng.random()
            if op_random < cx_prob:
                aspirant1, aspirant2 = toolbox.mate(
                    *map(toolbox.clone, toolbox.select(population, 2))
                )
                del aspirant1.fitness.values, aspirant2.fitness.values
                if accept_func(len(aspirant1)):
                    produced_pop.append(aspirant1)
                    produced_pop_sizes.append(len(aspirant1))

                if len(produced_pop) < count and accept_func(len(aspirant2)):
                    produced_pop.append(aspirant2)
                    produced_pop_sizes.append(len(aspirant2))
            else:
                aspirant = toolbox.clone(toolbox.select(population, 1)[0])
                if op_random - cx_prob < mut_prob:
                    aspirant = toolbox.mutate(aspirant)[0]
                    del aspirant.fitness.values
                if accept_func(len(aspirant)):
                    produced_pop.append(aspirant)
                    produced_pop_sizes.append(len(aspirant))

    return produced_pop, produced_pop_sizes


def _evaluate_invalid(toolbox: Toolbox, individuals: list[GPIndividual]) -> int:
    """Evaluate the individuals whose fitness is invalid.

    Args:
        toolbox: Toolbox with the evaluate and map operators.
        individuals: Individuals to scan for invalid fitness.

    Returns:
        The number of individuals that were evaluated.
    """
    invalid_ind = [ind for ind in individuals if not ind.fitness.is_valid()]
    fitness = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitness, strict=False):
        ind.fitness.values = fit
    return len(invalid_ind)


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

    nevals = _evaluate_invalid(toolbox, population)

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
        cutoff_size = _cutoff_size(natural_pop, pop_len, rho, min_cutoff)

        def target_prob(size: int, cutoff: int = cutoff_size, length: int = pop_len) -> float:
            return _target_prob(size, alpha, beta, gamma, length, cutoff)

        target_hist = _target_histogram(natural_hist, cutoff_size, target_prob)
        accept_func = _acceptance(natural_hist, target_hist, target_prob)

        offspring, _ = _produce(
            toolbox, population, pop_len, cx_prob, mut_prob, natural_pop, accept_func
        )

        nevals = _evaluate_invalid(toolbox, offspring)

        if hof is not None:
            hof.update(offspring)

        population[:] = offspring
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=nevals, **record)

        if verbose:
            print(logbook.stream)

    return population, logbook
