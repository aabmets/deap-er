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

from math import log, sqrt
from typing import Any

import numpy

__all__ = ["apply_cma_hyperparams"]


def apply_cma_hyperparams(strategy: Any, kwargs: dict[str, Any], *, rank_scale: float = 1.0) -> None:
    """Set λ, μ, weights, and CMA learning rates on ``strategy``.

    Default ``rank_one`` and ``rank_mu`` are multiplied by ``rank_scale``
    when those keys are omitted. Explicit kwargs are left unscaled.

    Args:
        strategy: CMA strategy with a ``dim`` attribute.
        kwargs: Constructor or ``compute_params`` keyword arguments.
        rank_scale: Multiplier for default rank-one and rank-μ rates.

    Raises:
        RuntimeError: If ``weights`` is not ``superlinear``,
            ``linear``, or ``equal``.
    """
    dim = strategy.dim
    default = int(4 + 3 * log(dim))
    strategy.lamb = int(kwargs.get("offsprings", default))
    strategy.mu = int(kwargs.get("survivors", int(strategy.lamb / 2)))
    scheme = kwargs.get("weights", "superlinear")
    if scheme == "superlinear":
        weights = log(strategy.mu + 0.5) - numpy.log(numpy.arange(1, strategy.mu + 1))
    elif scheme == "linear":
        weights = strategy.mu + 0.5 - numpy.arange(1, strategy.mu + 1)
    elif scheme == "equal":
        weights = numpy.ones(strategy.mu)
    else:
        raise RuntimeError(f"Unknown weights : {scheme}")
    strategy.weights = numpy.asarray(weights, dtype=float)
    strategy.weights /= sum(strategy.weights)
    strategy.mu_eff = 1.0 / sum(strategy.weights**2)
    default_one = 2.0 / ((dim + 1.3) ** 2 + strategy.mu_eff)
    if "rank_one" in kwargs:
        strategy.rank_one = float(kwargs["rank_one"])
    else:
        strategy.rank_one = float(default_one * rank_scale)
    temp_1 = strategy.mu_eff - 2.0 + 1.0 / strategy.mu_eff
    default_mu = 2.0 * temp_1 / ((dim + 2.0) ** 2 + strategy.mu_eff)
    if "rank_mu" in kwargs:
        strategy.rank_mu = float(kwargs["rank_mu"])
    else:
        strategy.rank_mu = float(default_mu * rank_scale)
    strategy.rank_mu = min(1 - strategy.rank_one, strategy.rank_mu)
    default = (strategy.mu_eff + 2.0) / (dim + strategy.mu_eff + 3.0)
    strategy.ss_cum = float(kwargs.get("ss_cum", default))
    temp_1 = sqrt((strategy.mu_eff - 1.0) / (dim + 1.0))
    default = 1.0 + 2.0 * max(0.0, temp_1 - 1.0) + strategy.ss_cum
    strategy.ss_dmp = float(kwargs.get("ss_dmp", default))
    strategy.cm_cum = float(kwargs.get("cm_cum", 4.0 / (dim + 4.0)))
