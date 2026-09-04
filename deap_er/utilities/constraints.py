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
from deap_er.base.dtypes import *
from collections.abc import Sequence, Callable
from itertools import repeat
from functools import wraps

__all__ = ["DeltaPenalty", "ClosestValidPenalty"]


class DeltaPenalty:
    """Decorator that penalizes fitness of invalid individuals.

    Valid individuals keep the original fitness. Invalid ones receive
    ``delta`` plus an optional distance penalty that grows as the
    individual moves away from the valid region.

    Args:
        feasibility: Function that reports whether an individual is valid.
        delta: Constant or sequence of constants used as the base
            penalty for an invalid individual.
        distance: Optional function returning the distance between the
            individual and a valid point.
    """

    def __init__(self, feasibility: Callable, delta: NumOrSeq, distance: Callable = None):
        """See the class docstring."""
        self.fea_func = feasibility
        if not isinstance(delta, Sequence):
            self.delta = repeat(delta)
        else:
            self.delta = delta
        self.dist_fct = distance

    def __call__(self, func):
        """Wrap a fitness function with the delta penalty.

        Args:
            func: Fitness function to decorate.

        Returns:
            A callable that returns the original fitness for valid
            individuals and the penalized fitness otherwise.
        """

        @wraps(func)
        def wrapper(individual, *args, **kwargs):
            if self.fea_func(individual):
                return func(individual, *args, **kwargs)

            weights = tuple(1 if w >= 0 else -1 for w in individual.fitness.weights)

            dists = [0 for _ in individual.fitness.weights]
            if self.dist_fct is not None:
                dists = self.dist_fct(individual)
                if not isinstance(dists, Sequence):
                    dists = repeat(dists)

            return tuple(d - w * dist for d, w, dist in zip(self.delta, weights, dists))

        return wrapper


class ClosestValidPenalty:
    """Decorator that penalizes fitness of invalid individuals.

    Valid individuals keep the original fitness. Invalid ones receive
    the fitness of the closest valid individual plus an optional
    weighted distance penalty that grows as the individual moves
    away from the valid region.

    Args:
        validity: Function that reports whether an individual is valid.
        feasible: Function that returns the closest feasible individual
            for an invalid one.
        alpha: Multiplication factor on the distance between the valid
            and invalid individuals.
        distance: Optional function returning the distance between the
            individual and a valid point.
    """

    def __init__(
        self, validity: Callable, feasible: Callable, alpha: float, distance: Callable = None
    ):
        """See the class docstring."""
        self.fea_func = validity
        self.fbl_fct = feasible
        self.alpha = alpha
        self.dist_fct = distance

    def __call__(self, func):
        """Wrap a fitness function with the closest-valid penalty.

        Args:
            func: Fitness function to decorate.

        Returns:
            A callable that returns the original fitness for valid
            individuals and the penalized fitness otherwise.
        """

        @wraps(func)
        def wrapper(individual, *args, **kwargs):
            if self.fea_func(individual):
                return func(individual, *args, **kwargs)

            f_ind = self.fbl_fct(individual)
            f_fbl = func(f_ind, *args, **kwargs)

            weights = tuple(1.0 if w >= 0 else -1.0 for w in individual.fitness.weights)

            if len(weights) != len(f_fbl):
                raise IndexError("Fitness weights and computed fitness are of different size.")
            dists = [0 for _ in individual.fitness.weights]
            if self.dist_fct is not None:
                dists = self.dist_fct(f_ind, individual)
                if not isinstance(dists, Sequence):
                    dists = repeat(dists)

            return tuple(f - w * self.alpha * d for f, w, d in zip(f_fbl, weights, dists))

        return wrapper
