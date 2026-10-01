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

from collections.abc import Iterable
from typing import Any

__all__: list[str] = [
    "CMA_HYPERPARAMS",
    "MO_HYPERPARAMS",
    "ONE_PLUS_LAMBDA_HYPERPARAMS",
    "merge_hyperparams",
]

CMA_HYPERPARAMS = (
    "offsprings",
    "survivors",
    "weights",
    "rank_one",
    "rank_mu",
    "ss_cum",
    "ss_dmp",
    "cm_cum",
)
ONE_PLUS_LAMBDA_HYPERPARAMS = (
    "offsprings",
    "thresh_sr",
    "ss_dmp",
    "tgt_sr",
    "ss_learn_rate",
    "th_cum",
    "cm_learn_rate",
)
MO_HYPERPARAMS = (*ONE_PLUS_LAMBDA_HYPERPARAMS, "survivors")


def merge_hyperparams(
    strategy: Any, kwargs: dict[str, Any], keys: Iterable[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge this call's hyperparameters over the ones pinned earlier.

    A hyperparameter given to the constructor or to any later
    ``compute_params`` / ``reset_state`` call stays pinned until a
    later call overrides it, the same way box bounds are kept. Only
    keys in ``keys`` are pinned; ``cm_init`` and box bounds are not.

    Args:
        strategy: Strategy whose ``hyperparams`` dict holds the pins.
        kwargs: This call's keyword arguments.
        keys: Hyperparameter names the strategy accepts.

    Returns:
        The kwargs to apply (pins with this call's kwargs on top) and
        the updated pins. Store the pins on ``strategy.hyperparams``
        only after they have been applied successfully.
    """
    pinned = dict(getattr(strategy, "hyperparams", {}))
    pinned.update({key: kwargs[key] for key in keys if key in kwargs})
    return {**kwargs, **pinned}, pinned
