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

import numpy

__all__: list[str] = [
    "CMA_UPDATED_STATE",
    "check_cma_state",
    "restore_cma_state",
    "snapshot_cma_state",
]

CMA_UPDATED_STATE = ("centroid", "sigma", "ps", "pc", "big_c")


def snapshot_cma_state(strategy: Any) -> dict[str, Any]:
    """Return the state an ``update`` call rebinds, for a rollback."""
    return {name: getattr(strategy, name) for name in CMA_UPDATED_STATE}


def restore_cma_state(strategy: Any, snapshot: dict[str, Any]) -> None:
    """Put back the state that ``snapshot_cma_state`` recorded."""
    for name, value in snapshot.items():
        setattr(strategy, name, value)


def check_cma_state(strategy: Any, snapshot: dict[str, Any]) -> None:
    """Roll back and raise when an update left a non-finite state.

    The state is non-finite when ``sigma`` is not a positive finite
    number, or any of ``centroid``, ``ps``, ``pc``, ``big_c`` holds
    a NaN or infinity. This happens once the step size collapses to
    zero or overflows.

    Args:
        strategy: CMA strategy after its centroid, paths, step size,
            and covariance were updated.
        snapshot: The state from ``snapshot_cma_state`` before the update.

    Raises:
        FloatingPointError: If the updated state is non-finite. Every
            attribute in ``snapshot`` is restored first.
    """
    sigma = float(strategy.sigma)
    arrays = (getattr(strategy, name) for name in ("centroid", "ps", "pc", "big_c"))
    if 0.0 < sigma < numpy.inf and all(numpy.isfinite(arr).all() for arr in arrays):
        return
    restore_cma_state(strategy, snapshot)
    raise FloatingPointError(
        f"CMA-ES update gave a non-finite state (sigma={sigma!r}); "
        "the strategy was left as it was before the update. Restart it "
        "with reset_state."
    )
