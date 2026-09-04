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
import bisect
from collections import defaultdict
from collections.abc import Callable, Sequence
from operator import itemgetter
from typing import Any, Literal, overload

from deap_er.base.dtypes import Individual

__all__ = ["sort_log_non_dominated"]


@overload
def sort_log_non_dominated(
    individuals: list[Individual], sel_count: int, ffo: Literal[True]
) -> list[Individual]: ...


@overload
def sort_log_non_dominated(
    individuals: list[Individual], sel_count: int, ffo: Literal[False] = False
) -> list[list[Individual]]: ...


def sort_log_non_dominated(
    individuals: list[Individual], sel_count: int, ffo: bool = False
) -> list[list[Individual]] | list[Individual]:
    """Sort individuals into non-dominated Pareto fronts.

    Uses the Generalized Reduced Run-Time Complexity Non-Dominated
    Sorting Algorithm.

    Args:
        individuals: Individuals to sort.
        sel_count: Number of individuals to select.
        ffo: If True, return only the first front. Optional.

    Returns:
        A list of Pareto fronts. The first element is the true
        Pareto front. An empty list if ``sel_count`` is 0. When
        ``ffo`` is True, the first front itself is returned.
    """
    if sel_count == 0:
        return []

    unique_fits = defaultdict(list)
    for ind in individuals:
        unique_fits[ind.fitness.wvalues].append(ind)

    obj = len(individuals[0].fitness.wvalues) - 1
    fitness = list(unique_fits.keys())
    front = dict.fromkeys(fitness, 0)

    fitness.sort(reverse=True)
    _sorting_helper_1(fitness, obj, front)

    nb_fronts = max(front.values()) + 1
    pareto_fronts = [[] for _ in range(nb_fronts)]
    for fit in fitness:
        index = front[fit]
        pareto_fronts[index].extend(unique_fits[fit])

    if not ffo:
        count = 0
        for i, front in enumerate(pareto_fronts):
            count += len(front)
            if count >= sel_count:
                return pareto_fronts[: i + 1]
        return pareto_fronts
    else:
        return pareto_fronts[0]


def _is_dominated(wvalues1: Sequence[Any], wvalues2: Sequence[Any]) -> bool:
    """Return whether ``wvalues1`` is strictly Pareto-dominated by ``wvalues2``.

    Both sequences are weighted objective values. Higher is better.

    Args:
        wvalues1: Candidate weighted values.
        wvalues2: Weighted values tested as a dominator.

    Returns:
        True if ``wvalues2`` is at least as good on every objective
        and strictly better on at least one.
    """
    not_equal = False
    for self_wvalue, other_wvalue in zip(wvalues1, wvalues2, strict=False):
        if self_wvalue > other_wvalue:
            return False
        elif self_wvalue < other_wvalue:
            not_equal = True
    return not_equal


def _median(seq: Sequence[Any], key: Callable[..., Any] | None = None) -> Any:
    """Return the median of ``seq``, optionally after applying ``key``.

    For an even-length sequence the two central values are averaged.

    Args:
        seq: Values to summarize.
        key: Optional transform applied before comparing and averaging.

    Returns:
        The median value.
    """
    key = key if key else lambda x: x
    sorted_seq = sorted(seq, key=key)
    length = len(seq)
    if length % 2 == 1:
        return key(sorted_seq[(length - 1) // 2])
    else:
        temp1 = key(sorted_seq[(length - 1) // 2])
        temp2 = key(sorted_seq[length // 2])
        return (temp1 + temp2) / 2.0


def _splitter(
    seq: Sequence[Any], obj: int, median: float
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Partition fitness vectors around ``median`` on objective ``obj``.

    Args:
        seq: Fitness vectors (weighted-value tuples).
        obj: Objective index used for the split.
        median: Threshold on that objective.

    Returns:
        Four lists: values at or above the median, values below,
        values above, and values at or below.
    """
    seq_1, seq_2, seq_3, seq_4 = [], [], [], []

    for fit in seq:
        if fit[obj] > median:
            seq_1.append(fit)
            seq_3.append(fit)
        elif fit[obj] < median:
            seq_2.append(fit)
            seq_4.append(fit)
        else:
            seq_1.append(fit)
            seq_4.append(fit)

    return seq_1, seq_2, seq_3, seq_4


def _sorting_helper_1(fitness: Sequence[Any], obj: int, front: dict[Any, int]) -> None:
    """Assign non-dominated front ranks on the first ``obj + 1`` objectives.

    Mutates ``front`` in place.

    Args:
        fitness: Unique fitness vectors, sorted best-first.
        obj: Highest objective index still under consideration.
        front: Map from fitness vector to front rank.
    """
    if len(fitness) < 2:
        return
    elif len(fitness) == 2:
        s1, s2 = fitness[0], fitness[1]
        if _is_dominated(s2[: obj + 1], s1[: obj + 1]):
            front[s2] = max(front[s2], front[s1] + 1)
    elif obj == 1:
        _sweep_a(fitness, front)
    elif len(frozenset(list(map(itemgetter(obj), fitness)))) == 1:
        _sorting_helper_1(fitness, obj - 1, front)
    else:
        best, worst = _split_a(fitness, obj)
        _sorting_helper_1(best, obj, front)
        _sorting_helper_2(best, worst, obj - 1, front)
        _sorting_helper_1(worst, obj, front)


def _split_a(fitness: Sequence[Any], obj: int) -> tuple[list[Any], list[Any]]:
    """Split ``fitness`` into a better half and a worse half on objective ``obj``.

    Chooses the more balanced of two median-split conventions.

    Args:
        fitness: Fitness vectors to partition.
        obj: Objective index used for the split.

    Returns:
        The better subset and the worse subset.
    """
    median_ = _median(fitness, itemgetter(obj))
    best_a, worst_a, best_b, worst_b = _splitter(fitness, obj, median_)

    balance_a = abs(len(best_a) - len(worst_a))
    balance_b = abs(len(best_b) - len(worst_b))

    if balance_a <= balance_b:
        return best_a, worst_a
    else:
        return best_b, worst_b


def _sweep_a(fitness: Sequence[Any], front: dict[Any, int]) -> None:
    """Update front ranks of a two-objective, already-sorted fitness list.

    Args:
        fitness: Fitness vectors ordered on the first objective.
        front: Map from fitness vector to front rank.
    """
    stairs = [-fitness[0][1]]
    f_stairs = [fitness[0]]
    for fit in fitness[1:]:
        idx = bisect.bisect_right(stairs, -fit[1])
        if 0 < idx <= len(stairs):
            f_stair = max(f_stairs[:idx], key=front.__getitem__)
            front[fit] = max(front[fit], front[f_stair] + 1)
        for i, f_stair in enumerate(f_stairs[idx:], idx):
            if front[f_stair] == front[fit]:
                del stairs[i]
                del f_stairs[i]
                break
        stairs.insert(idx, -fit[1])
        f_stairs.insert(idx, fit)


def _sorting_helper_2(
    best: Sequence[Any], worst: Sequence[Any], obj: int, front: dict[Any, int]
) -> None:
    """Raise front ranks in ``worst`` using domination from ``best``.

    Considers the first ``obj + 1`` objectives. Mutates ``front`` in
    place.

    Args:
        best: Fitness vectors already ranked.
        worst: Fitness vectors that may be dominated by ``best``.
        obj: Highest objective index still under consideration.
        front: Map from fitness vector to front rank.
    """
    key = itemgetter(obj)
    if len(worst) == 0 or len(best) == 0:
        return
    elif len(best) == 1 or len(worst) == 1:
        for hi in worst:
            for li in best:
                cond_1 = _is_dominated(hi[: obj + 1], li[: obj + 1])
                cond_2 = hi[: obj + 1] == li[: obj + 1]
                if cond_1 or cond_2:
                    front[hi] = max(front[hi], front[li] + 1)
    elif obj == 1:
        _sweep_b(best, worst, front)
    elif key(min(best, key=key)) >= key(max(worst, key=key)):
        _sorting_helper_2(best, worst, obj - 1, front)
    elif key(max(best, key=key)) >= key(min(worst, key=key)):
        best1, best2, worst1, worst2 = _split_b(best, worst, obj)
        _sorting_helper_2(best1, worst1, obj, front)
        _sorting_helper_2(best1, worst2, obj - 1, front)
        _sorting_helper_2(best2, worst2, obj, front)


def _split_b(
    best: Sequence[Any], worst: Sequence[Any], obj: int
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Split ``best`` and ``worst`` around a shared median on objective ``obj``.

    Chooses the more balanced of two split conventions.

    Args:
        best: Better fitness vectors.
        worst: Worse fitness vectors.
        obj: Objective index used for the split.

    Returns:
        Four subsets: high and low halves of ``best``, then of
        ``worst``.
    """
    median_ = _median(best) if len(best) > len(worst) else _median(worst, itemgetter(obj))

    best1_a, best2_a, best1_b, best2_b = _splitter(best, obj, median_)
    worst1_a, worst2_a, worst1_b, worst2_b = _splitter(worst, obj, median_)

    balance_a = abs(len(best1_a) - len(best2_a) + len(worst1_a) - len(worst2_a))
    balance_b = abs(len(best1_b) - len(best2_b) + len(worst1_b) - len(worst2_b))

    if balance_a <= balance_b:
        return best1_a, best2_a, worst1_a, worst2_a
    else:
        return best1_b, best2_b, worst1_b, worst2_b


def _sweep_b(best: Sequence[Any], worst: Sequence[Any], front: dict[Any, int]) -> None:
    """Update front ranks of ``worst`` from a two-objective sweep over ``best``.

    Args:
        best: Already-ranked fitness vectors, ordered on the first
            objective.
        worst: Fitness vectors whose ranks may increase.
        front: Map from fitness vector to front rank.
    """
    stairs, f_stairs = [], []
    iter_best = iter(best)
    next_best = next(iter_best, False)
    for h in worst:
        while next_best and h[:2] <= next_best[:2]:
            insert = True
            for i, f_stair in enumerate(f_stairs):
                if front[f_stair] == front[next_best]:
                    if f_stair[1] > next_best[1]:
                        insert = False
                    else:
                        del stairs[i], f_stairs[i]
                    break
            if insert:
                idx = bisect.bisect_right(stairs, -next_best[1])
                stairs.insert(idx, -next_best[1])
                f_stairs.insert(idx, next_best)
            next_best = next(iter_best, False)

        idx = bisect.bisect_right(stairs, -h[1])
        if 0 < idx <= len(stairs):
            f_stair = max(f_stairs[:idx], key=front.__getitem__)
            front[h] = max(front[h], front[f_stair] + 1)
