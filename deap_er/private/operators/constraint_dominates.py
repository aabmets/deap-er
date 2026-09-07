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

import math
from collections.abc import Callable
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
from deap_er.private.various.sort_non_dominated import sort_non_dominated

__all__: list[str] = ["constraint_dominates", "sort_constraint_dominated"]


def _require_constraint_callables(
    feasible: Callable[[Individual], bool] | None,
    violation: Callable[[Individual], float] | None,
) -> None:
    """Reject missing or non-callable constraint arguments.

    Args:
        feasible: Optional feasibility predicate.
        violation: Optional constraint-violation function.

    Raises:
        TypeError: If both are omitted, or a given argument is not
            callable.
    """
    if feasible is None and violation is None:
        raise TypeError("provide feasible or violation")
    if feasible is not None and not callable(feasible):
        raise TypeError("feasible must be a callable")
    if violation is not None and not callable(violation):
        raise TypeError("violation must be a callable")


def _as_violation(value: object) -> float:
    """Convert a caller-supplied violation to a finite scalar.

    Args:
        value: Value returned by the violation callable.

    Returns:
        The violation as a float.

    Raises:
        TypeError: If ``value`` is not a real scalar.
        ValueError: If ``value`` is not finite.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("violation must return a real scalar")
    cv = float(value)
    if not math.isfinite(cv):
        raise ValueError("violation must be a finite number")
    return cv


def _is_feasible(
    individual: Individual,
    feasible: Callable[[Individual], bool] | None,
    violation: Callable[[Individual], float] | None,
) -> bool:
    """Report whether ``individual`` is feasible under the caller rule.

    Args:
        individual: Individual to classify.
        feasible: Optional feasibility predicate. Wins when given.
        violation: Optional constraint-violation function. Used when
            ``feasible`` is omitted; ``<= 0`` is feasible.

    Returns:
        True if the individual is feasible.
    """
    if feasible is not None:
        return bool(feasible(individual))
    if violation is None:
        raise TypeError("provide feasible or violation")
    return _as_violation(violation(individual)) <= 0.0


def _infeasible_fronts(
    infeasible: list[Individual],
    violation: Callable[[Individual], float] | None,
) -> list[list[Individual]]:
    """Group infeasible individuals into Deb constraint fronts.

    Args:
        infeasible: Infeasible individuals in encounter order.
        violation: Optional constraint-violation function. Omitted
            means every infeasible shares one front.

    Returns:
        Fronts in increasing violation. Equal violations stay together.
    """
    if not infeasible:
        return []
    if violation is None:
        return [infeasible]
    keyed = [(_as_violation(violation(ind)), ind) for ind in infeasible]
    keyed.sort(key=lambda item: item[0])
    fronts: list[list[Individual]] = []
    current_cv = keyed[0][0]
    current: list[Individual] = []
    for cv, ind in keyed:
        if current and cv != current_cv:
            fronts.append(current)
            current = [ind]
            current_cv = cv
        else:
            current.append(ind)
    fronts.append(current)
    return fronts


def constraint_dominates(
    ind1: Individual,
    ind2: Individual,
    *,
    feasible: Callable[[Individual], bool] | None = None,
    violation: Callable[[Individual], float] | None = None,
) -> bool:
    """Return whether ``ind1`` constrained-dominates ``ind2``.

    Deb's NSGA-II rule (2002, §III-A): a feasible individual beats an
    infeasible one; two feasibles use ordinary Pareto dominance on
    ``fitness``; two infeasibles prefer the smaller constraint
    violation. Fitness values are not rewritten.

    When only ``violation`` is given, ``<= 0`` is feasible. When only
    ``feasible`` is given, infeasibles do not dominate each other.
    When both are given, the flag decides feasibility and the
    violation is used only among infeasibles.

    Args:
        ind1: Candidate that may dominate.
        ind2: Candidate that may be dominated.
        feasible: Predicate that reports whether an individual is
            feasible. Optional if ``violation`` is given.
        violation: Function returning a scalar constraint violation.
            Optional if ``feasible`` is given.

    Returns:
        True if ``ind1`` constrained-dominates ``ind2``.

    Raises:
        TypeError: If both callables are omitted, a given argument is
            not callable, or ``violation`` does not return a real
            scalar.
        ValueError: If ``violation`` returns a non-finite number.
    """
    _require_constraint_callables(feasible, violation)
    left = _is_feasible(ind1, feasible, violation)
    right = _is_feasible(ind2, feasible, violation)
    if left != right:
        return left
    if left:
        return bool(ind1.fitness.dominates(ind2.fitness))
    if violation is None:
        return False
    return _as_violation(violation(ind1)) < _as_violation(violation(ind2))


def sort_constraint_dominated(
    individuals: list[Individual],
    sel_count: int,
    *,
    feasible: Callable[[Individual], bool] | None = None,
    violation: Callable[[Individual], float] | None = None,
) -> list[list[Individual]]:
    """Sort individuals into Deb constrained-domination fronts.

    Every feasible individual constrained-dominates every infeasible
    one, so feasible Pareto fronts come first. Infeasibles follow in
    increasing violation; equal violations share a front. Fitness
    values are not rewritten.

    Args:
        individuals: Individuals to sort.
        sel_count: Number of individuals to place into fronts.
        feasible: Predicate that reports whether an individual is
            feasible. Optional if ``violation`` is given.
        violation: Function returning a scalar constraint violation.
            Optional if ``feasible`` is given.

    Returns:
        A list of fronts. The first element is the best constrained
        front. An empty list if ``sel_count`` is 0. A single empty
        front if ``individuals`` is empty and ``sel_count`` is
        positive.

    Raises:
        TypeError: If both callables are omitted, a given argument is
            not callable, or ``violation`` does not return a real
            scalar.
        ValueError: If ``violation`` returns a non-finite number.
    """
    if sel_count == 0:
        return []
    if not individuals:
        return [[]]
    _require_constraint_callables(feasible, violation)

    feasible_inds: list[Individual] = []
    infeasible_inds: list[Individual] = []
    for ind in individuals:
        if _is_feasible(ind, feasible, violation):
            feasible_inds.append(ind)
        else:
            infeasible_inds.append(ind)

    fronts: list[list[Individual]] = []
    placed = 0
    if feasible_inds:
        need = min(sel_count, len(feasible_inds))
        fronts.extend(sort_non_dominated(feasible_inds, need))
        placed = sum(len(front) for front in fronts)
        if placed >= sel_count:
            return fronts

    for front in _infeasible_fronts(infeasible_inds, violation):
        fronts.append(front)
        placed += len(front)
        if placed >= sel_count:
            break
    return fronts
