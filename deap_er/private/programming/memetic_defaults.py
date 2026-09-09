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

from typing import Any

__all__: list[str] = [
    "MEMETIC_DEFAULT_N_GEN",
    "MEMETIC_MAX_N_GEN",
    "cap_tune_n_gen",
    "estimate_tune_ephemerals_evals",
]

MEMETIC_DEFAULT_N_GEN = 2
MEMETIC_MAX_N_GEN = 5


def estimate_tune_ephemerals_evals(strategy: Any, n_gen: int) -> int:
    """Estimate how many evaluations a memetic tune would spend.

    Args:
        strategy: ``Strategy`` or ``StrategySeparable`` whose
            ``offsprings`` or ``lamb`` sets the batch size.
        n_gen: Inner CMA generations.

    Returns:
        ``n_gen`` times the per-generation offspring count.
    """
    offsprings = getattr(strategy, "offsprings", None)
    if offsprings is None:
        offsprings = getattr(strategy, "lamb", 1)
    return int(n_gen) * int(offsprings)


def cap_tune_n_gen(
    strategy: Any,
    n_gen: int,
    *,
    n_evals: int | None = None,
    nevals_used: int = 0,
    max_n_gen: int = MEMETIC_MAX_N_GEN,
) -> int:
    """Cap inner tune generations to a memetic leash and optional budget.

    Args:
        strategy: ``Strategy`` passed to
            :func:`~deap_er.gp.tune_ephemerals`.
        n_gen: Requested inner generations.
        n_evals: Optional evaluation budget. When set, the return
            value never exceeds what ``nevals_used`` can still afford.
        nevals_used: Evaluations already charged to the run.
        max_n_gen: Hard ceiling on inner generations. Defaults to
            :data:`MEMETIC_MAX_N_GEN`.

    Returns:
        A non-negative generation count. ``0`` means tune should no-op.

    Raises:
        ValueError: If ``nevals_used`` is negative or ``max_n_gen`` is
            less than ``1``.
    """
    if nevals_used < 0:
        raise ValueError("nevals_used must be at least 0")
    if max_n_gen < 1:
        raise ValueError("max_n_gen must be at least 1")
    capped = min(int(n_gen), max_n_gen)
    if n_evals is None:
        return capped
    remaining = int(n_evals) - nevals_used
    if remaining <= 0:
        return 0
    per_gen = estimate_tune_ephemerals_evals(strategy, 1)
    if per_gen <= 0:
        return 0
    return min(capped, remaining // per_gen)
