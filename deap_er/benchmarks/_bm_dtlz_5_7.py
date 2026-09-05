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
from functools import reduce
from math import cos, pi, sin

from deap_er.base.typedefs import Individual

__all__: list[str] = []


def bm_dtlz_5(individual: Individual, count: int) -> list[float]:
    r"""Evaluate the DTLZ5 multi-objective function.

    Returns a list of size ``count``. The individual must have at
    least ``count`` elements.

    Args:
        individual: Individual to evaluate.
        count: Number of objectives.

    Returns:
        Fitness values of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\mathbf{x}_m) = \text{ ?}`

       :math:`f_{1}(\mathbf{x}) = \text{ ?}`

       :math:`f_{2}(\mathbf{x}) = \text{ ?}`

       :math:`\ldots`

       :math:`f_{m}(\mathbf{x}) = \text{ ?}`


       Where :math:`m` is the number of objectives and :math:`\mathbf{x}_m`
       is a vector of the remaining attributes :math:`[x_m~\ldots~x_n]`
       of the individual in :math:`n > m` dimensions.
    """
    gval = sum([(a - 0.5) ** 2 for a in individual[count - 1 :]])
    return _dtlz_helper_2(individual, count, gval)


def bm_dtlz_6(individual: Individual, count: int) -> list[float]:
    r"""Evaluate the DTLZ6 multi-objective function.

    Returns a list of size ``count``. The individual must have at
    least ``count`` elements.

    Args:
        individual: Individual to evaluate.
        count: Number of objectives.

    Returns:
        Fitness values of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\mathbf{x}_m) = \text{ ?}`

       :math:`f_{1}(\mathbf{x}) = \text{ ?}`

       :math:`f_{2}(\mathbf{x}) = \text{ ?}`

       :math:`\ldots`

       :math:`f_{m}(\mathbf{x}) = \text{ ?}`


       Where :math:`m` is the number of objectives and :math:`\mathbf{x}_m`
       is a vector of the remaining attributes :math:`[x_m~\ldots~x_n]`
       of the individual in :math:`n > m` dimensions.
    """
    gval = sum([a**0.1 for a in individual[count - 1 :]])
    return _dtlz_helper_2(individual, count, gval)


def bm_dtlz_7(individual: Individual, count: int) -> list[float]:
    r"""Evaluate the DTLZ7 multi-objective function.

    Returns a list of size ``count``. The individual must have at
    least ``count`` elements.

    Args:
        individual: Individual to evaluate.
        count: Number of objectives.

    Returns:
        Fitness values of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\mathbf{x}_m) = \text{ ?}`

       :math:`f_{1}(\mathbf{x}) = \text{ ?}`

       :math:`f_{2}(\mathbf{x}) = \text{ ?}`

       :math:`\ldots`

       :math:`f_{m}(\mathbf{x}) = \text{ ?}`


       Where :math:`m` is the number of objectives and :math:`\mathbf{x}_m`
       is a vector of the remaining attributes :math:`[x_m~\ldots~x_n]`
       of the individual in :math:`n > m` dimensions.
    """

    def fn(a: float) -> float:
        return float(a / (1 + gval) * (1 + sin(3 * pi * a)))

    gval = sum(individual[count - 1 :])
    gval = 1 + 9 / len(individual[count - 1 :]) * gval

    fit = list(individual[: count - 1])
    vals = [fn(a) for a in individual[: count - 1]]
    res = (1 + gval) * (count - sum(vals))
    fit.append(res)
    return [float(value) for value in fit]


def _dtlz_helper_2(individual: Individual, count: int, gval: float) -> list[float]:
    """Build DTLZ5-style degenerate-front objectives.

    Args:
        individual: Decision vector.
        count: Number of objectives.
        gval: Distance-to-front term ``g``.

    Returns:
        Objective values of length ``count``.
    """

    def theta(x: float) -> float:
        return pi / (4.0 * (1 + gval)) * (1 + 2 * gval * x)

    vals = [cos(theta(a)) for a in individual[1 : count - 1]]
    rdc = reduce(lambda x, y: x * y, vals, 1)
    fit = [(1 + gval) * cos(pi / 2.0 * individual[0]) * rdc]

    for m in reversed(range(1, count)):
        if m == 1:
            res = (1 + gval) * sin(pi / 2 * individual[0])
        else:
            vals = [cos(theta(a)) for a in individual[1 : m - 1]]
            rdc = reduce(lambda x, y: x * y, vals, 1)
            _cos = cos(pi / 2 * individual[0])
            _sin = sin(theta(individual[m - 1]))
            res = (1 + gval) * _cos * rdc * _sin
        fit.append(res)
    return [float(value) for value in fit]
