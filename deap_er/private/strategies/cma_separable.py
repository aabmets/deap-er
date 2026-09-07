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

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

from .cma_params import (
    CmaCore,
    adapt_cma_sigma,
    apply_cma_hyperparams,
    generate_cma_offspring,
    init_cma_state,
    reset_cma_state,
    shift_cma_centroid,
    update_cma_paths,
)
from .common import update_bound_attrs

__all__ = ["StrategySeparable"]


class StrategySeparable(CmaCore):
    """Separable CMA-ES with a diagonal covariance (Ros and Hansen, 2008).

    Learns one variance per gene. Memory and the generate/update step
    are ``O(n)``. There is no learned correlation. Default
    ``rank_one`` and ``rank_mu`` are the ``Strategy`` defaults scaled
    by ``(n + 2) / 3``. Step-size uses the same cumulative step-size
    adaptation as ``Strategy``. Box bounds and the rest of the
    keyword surface match ``Strategy``, except ``cm_init`` is a
    length-``n`` variance vector (default ones), not an ``n``-by-``n``
    matrix.

    Args:
        centroid: Starting point of the search distribution.
        sigma: Initial standard deviation of the distribution.
        **kwargs: Optional strategy parameters. Shared names follow
            ``Strategy`` (``offsprings``, ``survivors``, ``weights``,
            ``cm_cum``, ``ss_cum``, ``ss_dmp``, ``rank_one``,
            ``rank_mu``, ``low``, ``up``, ``bound_mode``,
            ``resample_limit``). ``cm_init`` must be a length-``n``
            variance vector or a scalar broadcast to ``n``.

    Raises:
        RuntimeError: If ``weights`` is not ``superlinear``,
            ``linear``, or ``equal``.
        ValueError: If ``cm_init`` is not a length-``n`` vector, any
            variance is not positive, or box-bound kwargs are invalid.
    """

    def __init__(self, centroid: Iterable[float], sigma: float, **kwargs: Any) -> None:
        """See the class docstring."""
        init_cma_state(self, centroid, sigma)
        self.compute_params(**kwargs)

    def compute_params(self, **kwargs: Any) -> None:
        """Recompute λ, rates, and the diagonal ``cm_init`` vector.

        Args:
            **kwargs: Same names as ``Strategy.compute_params``, except
                ``cm_init`` is a length-``n`` variance vector.

        Raises:
            RuntimeError: If ``weights`` is unknown.
            ValueError: If ``cm_init`` is missing, the wrong shape, or
                not strictly positive.
        """
        apply_cma_hyperparams(self, kwargs, rank_scale=(self.dim + 2.0) / 3.0)
        if not hasattr(self, "big_c") or "cm_init" in kwargs:
            self.big_c = _variance_vector(kwargs.get("cm_init", numpy.ones(self.dim)), self.dim)
            self.diag_d = numpy.sqrt(self.big_c)
            self.cond = float(numpy.max(self.big_c) / numpy.min(self.big_c))
        update_bound_attrs(self, kwargs)

    def reset_state(
        self,
        centroid: Iterable[float],
        sigma: float,
        **kwargs: Any,
    ) -> None:
        """Reset mutable CMA state for a restart."""
        reset_cma_state(self, centroid, sigma)
        self.compute_params(cm_init=numpy.ones(self.dim), **kwargs)

    def generate(self, ind_init: Callable[..., Individual]) -> list[Individual]:
        """Draw ``lamb`` axis-aligned samples and apply box bounds.

        Args:
            ind_init: Builds an individual from a length-``n`` vector.

        Returns:
            The sampled population.
        """
        return generate_cma_offspring(self, self.diag_d, ind_init)

    def update(self, population: list[Individual]) -> None:
        """Update centroid, step-size, and diagonal covariance.

        Individuals are ranked by fitness. The best ``survivors``
        members drive the update.

        Args:
            population: Evaluated individuals from ``generate``.
        """
        old_centroid, c_diff = shift_cma_centroid(self, population)
        hsig = update_cma_paths(self, c_diff, c_diff / self.diag_d)
        ar_tmp = numpy.asarray(population[0 : self.mu]) - old_centroid
        decay = (1 - hsig) * self.rank_one * self.cm_cum * (2 - self.cm_cum)
        keep = 1 - self.rank_one - self.rank_mu + decay
        self.big_c = (
            keep * self.big_c
            + self.rank_one * self.pc**2
            + self.rank_mu * numpy.dot(self.weights, ar_tmp**2) / self.sigma**2
        )
        self.big_c = numpy.maximum(self.big_c, numpy.finfo(float).tiny)
        adapt_cma_sigma(self)
        self.diag_d = numpy.sqrt(self.big_c)
        self.cond = float(numpy.max(self.big_c) / numpy.min(self.big_c))
        self.update_count += 1


def _variance_vector(cm_init: Any, dim: int) -> numpy.ndarray:
    """Return a length-``dim`` positive variance vector from ``cm_init``."""
    values = numpy.asarray(cm_init, dtype=float)
    if values.ndim == 0:
        values = numpy.full(dim, float(values))
    if values.ndim != 1 or values.size != dim:
        raise ValueError("cm_init must be a length-n variance vector.")
    if numpy.any(values <= 0.0):
        raise ValueError("cm_init must contain only positive variances.")
    return values
