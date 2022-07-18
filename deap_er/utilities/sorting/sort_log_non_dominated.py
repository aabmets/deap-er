# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 - Mattias Aabmets, The DEAP Team and Other Contributors           #
#                                                                                        #
#   Permission is hereby granted, free of charge, to any person obtaining a copy         #
#   of this software and associated documentation files (the "Software"), to deal        #
#   in the Software without restriction, including without limitation the rights         #
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell            #
#   copies of the Software, and to permit persons to whom the Software is                #
#   furnished to do so, subject to the following conditions:                             #
#                                                                                        #
#   The above copyright notice and this permission notice shall be included in all       #
#   copies or substantial portions of the Software.                                      #
#                                                                                        #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR           #
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,             #
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE          #
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER               #
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,        #
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE        #
#   SOFTWARE.                                                                            #
#                                                                                        #
# ====================================================================================== #
from collections import defaultdict
from typing import Callable, Sequence
from operator import itemgetter
import bisect


__all__ = ['sort_log_non_dominated']


# ====================================================================================== #
def sort_log_non_dominated(individuals: list, sel_count: int,
                           first_front_only: bool = False) -> list:
    """
    Sorts *individuals* in pareto non-dominated fronts
    using the Generalized Reduced Run-Time Complexity
    Non-Dominated Sorting Algorithm.

    Parameters:
        individuals: A list of individuals to sort.
        sel_count: The number of individuals to select.
        first_front_only: If True, only the first front is returned.
    Returns:
        A list of Pareto fronts, where the first
        element is the true Pareto front.
    """
    if sel_count == 0:
        return []

    unique_fits = defaultdict(list)
    for i, ind in enumerate(individuals):
        unique_fits[ind.fitness.wvalues].append(ind)

    obj = len(individuals[0].fitness.wvalues)-1
    fitness = list(unique_fits.keys())
    front = dict.fromkeys(fitness, 0)

    fitness.sort(reverse=True)
    _sorting_helper_1(fitness, obj, front)

    nb_fronts = max(front.values())+1
    pareto_fronts = [[] for _ in range(nb_fronts)]
    for fit in fitness:
        index = front[fit]
        pareto_fronts[index].extend(unique_fits[fit])

    if not first_front_only:
        count = 0
        for i, front in enumerate(pareto_fronts):
            count += len(front)
            if count >= sel_count:
                return pareto_fronts[:i+1]
        return pareto_fronts
    else:
        return pareto_fronts[0]


# -------------------------------------------------------------------------------------- #
def _is_dominated(wvalues1: Sequence, wvalues2: Sequence) -> bool:
    not_equal = False
    for self_wvalue, other_wvalue in zip(wvalues1, wvalues2):
        if self_wvalue > other_wvalue:
            return False
        elif self_wvalue < other_wvalue:
            not_equal = True
    return not_equal


# -------------------------------------------------------------------------------------- #
def _median(seq: Sequence, key: Callable = None) -> float:
    key = key if key else lambda x: x
    sorted_seq = sorted(seq, key=key)
    length = len(seq)
    if length % 2 == 1:
        return key(sorted_seq[(length - 1) // 2])
    else:
        temp1 = key(sorted_seq[(length - 1) // 2])
        temp2 = key(sorted_seq[length // 2])
        return (temp1 + temp2) / 2.0


# -------------------------------------------------------------------------------------- #
def _splitter(seq: Sequence, obj: int, median: float) -> tuple:
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


# -------------------------------------------------------------------------------------- #
def _sorting_helper_1(fitness: Sequence, obj: int, front: dict) -> None:
    if len(fitness) < 2:
        return
    elif len(fitness) == 2:
        s1, s2 = fitness[0], fitness[1]
        if _is_dominated(s2[:obj + 1], s1[:obj + 1]):
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


# -------------------------------------------------------------------------------------- #
def _split_a(fitness: Sequence, obj: int):
    median_ = _median(fitness, itemgetter(obj))
    best_a, worst_a, best_b, worst_b = _splitter(fitness, obj, median_)

    balance_a = abs(len(best_a) - len(worst_a))
    balance_b = abs(len(best_b) - len(worst_b))

    if balance_a <= balance_b:
        return best_a, worst_a
    else:
        return best_b, worst_b


# -------------------------------------------------------------------------------------- #
def _sweep_a(fitness: Sequence, front: dict) -> None:
    stairs = [-fitness[0][1]]
    f_stairs = [fitness[0]]
    for fit in fitness[1:]:
        idx = bisect.bisect_right(stairs, -fit[1])
        if 0 < idx <= len(stairs):
            f_stair = max(f_stairs[:idx], key=front.__getitem__)
            front[fit] = max(front[fit], front[f_stair]+1)
        for i, f_stair in enumerate(f_stairs[idx:], idx):
            if front[f_stair] == front[fit]:
                del stairs[i]
                del f_stairs[i]
                break
        stairs.insert(idx, -fit[1])
        f_stairs.insert(idx, fit)


# -------------------------------------------------------------------------------------- #
def _sorting_helper_2(best: Sequence, worst: Sequence, obj: int, front: dict) -> None:
    key = itemgetter(obj)
    if len(worst) == 0 or len(best) == 0:
        return
    elif len(best) == 1 or len(worst) == 1:
        for hi in worst:
            for li in best:
                cond_1 = _is_dominated(hi[:obj + 1], li[:obj + 1])
                cond_2 = hi[:obj + 1] == li[:obj + 1]
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


# -------------------------------------------------------------------------------------- #
def _split_b(best: Sequence, worst: Sequence, obj: int):
    if len(best) > len(worst):
        median_ = _median(best)
    else:
        median_ = _median(worst, itemgetter(obj))

    best1_a, best2_a, best1_b, best2_b = _splitter(best, obj, median_)
    worst1_a, worst2_a, worst1_b, worst2_b = _splitter(worst, obj, median_)

    balance_a = abs(len(best1_a) - len(best2_a) + len(worst1_a) - len(worst2_a))
    balance_b = abs(len(best1_b) - len(best2_b) + len(worst1_b) - len(worst2_b))

    if balance_a <= balance_b:
        return best1_a, best2_a, worst1_a, worst2_a
    else:
        return best1_b, best2_b, worst1_b, worst2_b


# -------------------------------------------------------------------------------------- #
def _sweep_b(best, worst, front):
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
            front[h] = max(front[h], front[f_stair]+1)
