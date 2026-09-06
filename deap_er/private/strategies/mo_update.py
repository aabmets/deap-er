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
from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING, Any

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
from deap_er.private.various.least_contrib import least_contrib
from deap_er.private.various.sort_non_dominated import sort_non_dominated

from .common import step_size_multiplier

__all__: list[str] = [
    "select",
    "rank_one_update",
    "copy_offspring_state",
    "update_chosen_offspring",
    "decay_rejected_offspring",
    "commit_parent_params",
]


def select(
    strategy: Any, candidates: list[Individual]
) -> tuple[list[Individual], list[Individual]]:
    """Split candidates into ``survivors`` chosen and the remainder.

    Uses non-dominated sorting. When a front would overflow
    ``survivors``, extra members are dropped by least hypervolume
    contribution.

    Args:
        strategy: Multi-objective CMA strategy.
        candidates: Individuals to rank.

    Returns:
        Chosen individuals and those not selected.
    """
    if len(candidates) <= strategy.mu:
        return candidates, []

    pareto_fronts = sort_non_dominated(candidates, len(candidates))

    chosen: list[Individual] = []
    mid_front: list[Individual] = []
    not_chosen: list[Individual] = []

    full = False
    for front in pareto_fronts:
        if len(chosen) + len(front) <= strategy.mu and not full:
            chosen += front
        elif not mid_front and len(chosen) < strategy.mu:
            mid_front = front
            full = True
        else:
            not_chosen += front

    k = strategy.mu - len(chosen)
    if k > 0 and mid_front:
        ref = numpy.max(numpy.array([ind.fitness.wvalues for ind in candidates]) * -1, axis=0) + 1

        for _ in range(len(mid_front) - k):
            idx = least_contrib(mid_front, ref)
            not_chosen.append(mid_front.pop(idx))

        chosen += mid_front

    return chosen, not_chosen


def rank_one_update(
    inv_cholesky: numpy.ndarray,
    big_a: numpy.ndarray,
    alpha: float,
    beta: float,
    v: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Apply a rank-one covariance update to the Cholesky factors.

    Args:
        inv_cholesky: Inverse of the current Cholesky factor.
        big_a: Current Cholesky factor of the covariance.
        alpha: Weight of the existing covariance.
        beta: Weight of the rank-one term.
        v: Evolution-path vector used in the update.

    Returns:
        Updated inverse Cholesky factor and Cholesky factor.
    """
    w = numpy.dot(inv_cholesky, v)

    if float(numpy.max(numpy.abs(w))) > 1e-20:
        w_inv = numpy.dot(w, inv_cholesky)
        norm_w2 = numpy.sum(w**2)
        a = sqrt(alpha)
        root = numpy.sqrt(1 + beta / alpha * norm_w2)
        b = a / norm_w2 * (root - 1)
        big_a = a * big_a + b * numpy.outer(v, w)
        part = a**2 + a * b * norm_w2
        inv_cholesky = 1.0 / a * inv_cholesky - b / part * numpy.outer(w, w_inv)
    else:
        a = sqrt(alpha)
        big_a = a * big_a
        inv_cholesky = inv_cholesky / a

    return inv_cholesky, big_a


def copy_offspring_state(
    strategy: Any, chosen: list[Individual]
) -> tuple[list[Any], list[Any], list[Any], list[Any], list[Any], list[Any]]:
    """Copy CMA state from each chosen individual's parent.

    Offspring rows copy the parent at ``ps_[1]``. Surviving
    parents leave a None placeholder.

    Args:
        strategy: Multi-objective CMA strategy.
        chosen: Individuals kept as the next parent set.

    Returns:
        Parallel lists of last step-size, sigma, inverse
        Cholesky, Cholesky factor, evolution path, and success
        rate.
    """
    last_steps: list[Any] = []
    sigmas: list[Any] = []
    inv_cholesky: list[Any] = []
    big_a: list[Any] = []
    pc: list[Any] = []
    psucc: list[Any] = []
    bags = (last_steps, sigmas, inv_cholesky, big_a, pc, psucc)
    for ind in chosen:
        if ind.ps_[0] != "o":
            for bag in bags:
                bag.append(None)
            continue
        idx = ind.ps_[1]
        last_steps.append(strategy.sigmas[idx])
        sigmas.append(strategy.sigmas[idx])
        inv_cholesky.append(strategy.inv_cholesky[idx].copy())
        big_a.append(strategy.big_a[idx].copy())
        pc.append(strategy.pc[idx].copy())
        psucc.append(strategy.psucc[idx])
    return last_steps, sigmas, inv_cholesky, big_a, pc, psucc


def update_chosen_offspring(
    strategy: Any,
    chosen: list[Individual],
    last_steps: list[Any],
    sigmas: list[Any],
    inv_cholesky: list[Any],
    big_a: list[Any],
    pc: list[Any],
    psucc: list[Any],
) -> None:
    """Update copied CMA state for each successful offspring.

    Args:
        strategy: Multi-objective CMA strategy.
        chosen: Individuals kept as the next parent set.
        last_steps: Parent step-size at birth, per chosen slot.
        sigmas: Step-size accumulator, updated in place.
        inv_cholesky: Inverse Cholesky accumulator.
        big_a: Cholesky-factor accumulator.
        pc: Evolution-path accumulator.
        psucc: Success-rate accumulator.
    """
    cp, cc, c_cov = strategy.ss_learn_rate, strategy.th_cum, strategy.cm_learn_rate
    d, pt_arg, p_thresh = strategy.ss_dmp, strategy.tgt_sr, strategy.thresh_sr
    for i, ind in enumerate(chosen):
        t, p_idx = ind.ps_
        if t != "o":
            continue
        psucc[i] = (1.0 - cp) * psucc[i] + cp
        sigmas[i] = sigmas[i] * step_size_multiplier(psucc[i], pt_arg, d)
        if psucc[i] < p_thresh:
            xp = numpy.array(ind)
            x = numpy.array(strategy.parents[p_idx])
            pc[i] = (1.0 - cc) * pc[i] + sqrt(cc * (2.0 - cc)) * (xp - x) / last_steps[i]
            alpha = 1 - c_cov
        else:
            pc[i] = (1.0 - cc) * pc[i]
            alpha = 1 - c_cov + c_cov * cc * (2.0 - cc)
        inv_cholesky[i], big_a[i] = rank_one_update(inv_cholesky[i], big_a[i], alpha, c_cov, pc[i])


def decay_rejected_offspring(
    strategy: Any, not_chosen: list[Individual], chosen: list[Individual]
) -> None:
    """Apply one success-rate and step-size update per sampled parent.

    Igel/Voss uses one trial per generation: the success rate is
    ``n_selected / n_from_parent``, then ``p_succ`` and ``σ`` update
    once. Parallel children of one parent are not stacked.

    Args:
        strategy: Multi-objective CMA strategy.
        not_chosen: Individuals dropped by selection.
        chosen: Individuals kept as the next parent set.
    """
    cp, d, pt_arg = strategy.ss_learn_rate, strategy.ss_dmp, strategy.tgt_sr
    n_from: dict[int, int] = {}
    n_sel: dict[int, int] = {}
    for bag, inds in ((n_from, chosen + not_chosen), (n_sel, chosen)):
        for ind in inds:
            t, p_idx = ind.ps_
            if t == "o":
                bag[p_idx] = bag.get(p_idx, 0) + 1
    for p_idx, n in n_from.items():
        rate = n_sel.get(p_idx, 0) / n
        strategy.psucc[p_idx] = (1.0 - cp) * strategy.psucc[p_idx] + cp * rate
        strategy.sigmas[p_idx] *= step_size_multiplier(strategy.psucc[p_idx], pt_arg, d)


def commit_parent_params(
    strategy: Any,
    chosen: list[Individual],
    sigmas: list[Any],
    inv_cholesky: list[Any],
    big_a: list[Any],
    pc: list[Any],
    psucc: list[Any],
) -> None:
    """Write merged CMA parameters onto this strategy.

    Offspring slots take the updated copy. Surviving parents keep
    their existing parameter rows.

    Args:
        strategy: Multi-objective CMA strategy.
        chosen: Individuals kept as the next parent set.
        sigmas: Updated step sizes for offspring slots.
        inv_cholesky: Updated inverse Cholesky factors.
        big_a: Updated Cholesky factors.
        pc: Updated evolution paths.
        psucc: Updated success rates.
    """
    sources = {
        "inv_cholesky": inv_cholesky,
        "sigmas": sigmas,
        "big_a": big_a,
        "psucc": psucc,
        "pc": pc,
    }
    for name, var in sources.items():
        attr = getattr(strategy, name)
        merged = []
        for i, ind in enumerate(chosen):
            if ind.ps_[0] == "o":
                merged.append(var[i])
            else:
                merged.append(attr[ind.ps_[1]])
        setattr(strategy, name, merged)
