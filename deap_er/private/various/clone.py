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

from array import array
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = ["clone_individual"]


def clone_individual(individual: Individual) -> Individual:
    """Copy a sequence individual and its fitness without a full deepcopy.

    Register this on a Toolbox when a shallow gene copy is enough:
    ``toolbox.register("clone", tools.clone_individual)``. Falls back to
    ``copy.deepcopy`` when the individual is a NumPy array or carries
    extra state such as ``strategy``, ``ps_``, or ``history_index``.

    Args:
        individual: Individual to copy.

    Returns:
        An independent copy of ``individual``.
    """
    extra = getattr(individual, "__dict__", None)
    if extra is not None and extra.keys() - {"fitness"}:
        return deepcopy(individual)
    if hasattr(individual, "strategy") or hasattr(individual, "ps_"):
        return deepcopy(individual)
    if hasattr(individual, "history_index"):
        return deepcopy(individual)
    if isinstance(individual, numpy.ndarray):
        return deepcopy(individual)
    if not isinstance(individual, list | array):
        return deepcopy(individual)

    clone = type(individual)(individual)
    if hasattr(individual, "fitness"):
        clone.fitness = deepcopy(individual.fitness)
    return clone
