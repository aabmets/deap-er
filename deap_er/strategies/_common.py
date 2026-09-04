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
from collections.abc import Callable
from math import exp

import numpy

from deap_er.base.dtypes import Individual
from deap_er.rng import rng

__all__: list[str] = []


def _sample_offspring(
    center: numpy.ndarray | Individual,
    sigma: float,
    transform: numpy.ndarray,
    lamb: int,
    dim: int,
    ind_init: Callable[..., Individual],
) -> list[Individual]:
    """Sample individuals from a multivariate normal distribution.

    Draws ``lamb`` standard normal vectors and maps them through
    ``transform`` scaled by ``sigma``, around ``center``.

    Args:
        center: Mean of the search distribution.
        sigma: Standard deviation of the distribution.
        transform: Matrix that shapes the distribution, usually a
            Cholesky factor or a scaled eigenbasis.
        lamb: Number of individuals to sample.
        dim: Dimensionality of the search space.
        ind_init: Callable that turns a sampled vector into an
            individual.

    Returns:
        Newly sampled individuals.
    """
    arz = rng.standard_normal((lamb, dim))
    arz = center + sigma * numpy.dot(arz, transform.T)
    return list(map(ind_init, arz))


def _step_size_multiplier(psucc: float, tgt_sr: float, ss_dmp: float) -> float:
    """Return the step-size factor implied by a success rate.

    The step-size grows while the success rate is above ``tgt_sr`` and
    shrinks below it.

    Args:
        psucc: Smoothed success rate.
        tgt_sr: Target success rate.
        ss_dmp: Damping of the step-size.

    Returns:
        The multiplier to apply to the current step-size.
    """
    return exp((psucc - tgt_sr) / (ss_dmp * (1.0 - tgt_sr)))
