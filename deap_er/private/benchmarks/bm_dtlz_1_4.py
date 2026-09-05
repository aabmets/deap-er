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

from functools import reduce
from math import cos, pi, sin
from operator import mul
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = ["bm_dtlz_1", "bm_dtlz_2", "bm_dtlz_3", "bm_dtlz_4"]


def bm_dtlz_1(individual: Individual, count: int) -> list[float]:
    r"""Evaluate the DTLZ1 multi-objective function.

    Returns a list of size ``count``. The individual must have at
    least ``count`` elements.

    Args:
        individual: Individual to evaluate.
        count: Number of objectives.

    Returns:
        Fitness values of the individual.

    ??? note "Equations"

        $$
        g(\mathbf{x}_m) = 100\left(|\mathbf{x}_m| +
        \sum_{x_i \in \mathbf{x}_m}\left((x_i - 0.5)^2 -
        \cos(20\pi(x_i - 0.5))\right)\right)
        $$

        $f_{1}(\mathbf{x}) = \frac{1}{2} (1 + g(\mathbf{x}_m)) \prod_{i=1}^{m-1}x_i$

        $$
        f_{2}(\mathbf{x}) = \frac{1}{2} (1 + g(\mathbf{x}_m))
        (1-x_{m-1}) \prod_{i=1}^{m-2}x_i
        $$

        $f_{m-1}(\mathbf{x}) = \frac{1}{2} (1 + g(\mathbf{x}_m)) (1 - x_2) x_1$

        $\ldots$

        $f_{m}(\mathbf{x}) = \frac{1}{2} (1 - x_1)(1 + g(\mathbf{x}_m))$

        Where $m$ is the number of objectives and $\mathbf{x}_m$
        is a vector of the remaining attributes $[x_m~\ldots~x_n]$
        of the individual in $n > m$ dimensions.
    """

    def fn_xi(xi: float) -> float:
        _cos = cos(20 * pi * (xi - 0.5))
        return float((xi - 0.5) ** 2 - _cos)

    def fn_m(m: int) -> float:
        rdc = reduce(mul, individual[:m], 1)
        return float(0.5 * rdc * (1 - individual[m]) * (1 + gval))

    _sum = sum(fn_xi(xi) for xi in individual[count - 1 :])
    gval = 100 * (len(individual[count - 1 :]) + _sum)
    fit = [0.5 * reduce(mul, individual[: count - 1], 1) * (1 + gval)]
    fit.extend(fn_m(m) for m in reversed(range(count - 1)))
    return [float(value) for value in fit]


def bm_dtlz_2(individual: Individual, count: int) -> list[float]:
    r"""Evaluate the DTLZ2 multi-objective function.

    Returns a list of size ``count``. The individual must have at
    least ``count`` elements.

    Args:
        individual: Individual to evaluate.
        count: Number of objectives.

    Returns:
        Fitness values of the individual.

    ??? note "Equations"

        $g(\mathbf{x}_m) = \sum_{x_i \in \mathbf{x}_m} (x_i - 0.5)^2$

        $f_{1}(\mathbf{x}) = (1 + g(\mathbf{x}_m)) \prod_{i=1}^{m-1} \cos(0.5x_i\pi)$

        $$
        f_{2}(\mathbf{x}) = (1 + g(\mathbf{x}_m))
        \sin(0.5x_{m-1}\pi) \prod_{i=1}^{m-2}
        \cos(0.5x_i\pi)
        $$

        $\ldots$

        $f_{m}(\mathbf{x}) = (1 + g(\mathbf{x}_m)) \sin(0.5x_{1}\pi )$

        Where $m$ is the number of objectives and $\mathbf{x}_m$
        is a vector of the remaining attributes $[x_m~\ldots~x_n]$
        of the individual in $n > m$ dimensions.
    """
    xm = individual[count - 1 :]
    gval = sum((xi - 0.5) ** 2 for xi in xm)
    return _dtlz_helper_1(individual, count, gval)


def bm_dtlz_3(individual: Individual, count: int) -> list[float]:
    r"""Evaluate the DTLZ3 multi-objective function.

    Returns a list of size ``count``. The individual must have at
    least ``count`` elements.

    Args:
        individual: Individual to evaluate.
        count: Number of objectives.

    Returns:
        Fitness values of the individual.

    ??? note "Equations"

        $$
        g(\mathbf{x}_m) = 100\left(|\mathbf{x}_m| +
        \sum_{x_i \in \mathbf{x}_m}\left((x_i - 0.5)^2 -
        \cos(20\pi(x_i - 0.5))\right)\right)
        $$

        $f_{1}(\mathbf{x}) = (1 + g(\mathbf{x}_m)) \prod_{i=1}^{m-1} \cos(0.5x_i\pi)$

        $$
        f_{2}(\mathbf{x}) = (1 + g(\mathbf{x}_m))
        \sin(0.5x_{m-1}\pi) \prod_{i=1}^{m-2}
        \cos(0.5x_i\pi)
        $$

        $\ldots$

        $f_{m}(\mathbf{x}) = (1 + g(\mathbf{x}_m)) \sin(0.5x_{1}\pi )$

        Where $m$ is the number of objectives and $\mathbf{x}_m$
        is a vector of the remaining attributes $[x_m~\ldots~x_n]$
        of the individual in $n > m$ dimensions.
    """

    def fn(xi: float) -> float:
        _cos = cos(20 * pi * (xi - 0.5))
        return float((xi - 0.5) ** 2 - _cos)

    xm = individual[count - 1 :]
    gval = 100 * (len(xm) + sum(fn(xi) for xi in xm))
    return _dtlz_helper_1(individual, count, gval)


def bm_dtlz_4(individual: Individual, count: int, alpha: float) -> list[float]:
    r"""Evaluate the DTLZ4 multi-objective function.

    Returns a list of size ``count``. The individual must have at
    least ``count`` elements.

    Args:
        individual: Individual to evaluate.
        count: Number of objectives.
        alpha: Fitness values exponentiation factor.

    Returns:
        Fitness values of the individual.

    ??? note "Equations"

        $g(\mathbf{x}_m) = \sum_{x_i \in \mathbf{x}_m} (x_i - 0.5)^2$

        $f_{1}(\mathbf{x}) = (1 + g(\mathbf{x}_m)) \prod_{i=1}^{m-1} \cos(0.5x_i^\alpha\pi)$

        $$
        f_{2}(\mathbf{x}) = (1 + g(\mathbf{x}_m))
        \sin(0.5x_{m-1}^\alpha\pi)
        \prod_{i=1}^{m-2} \cos(0.5x_i^\alpha\pi)
        $$

        $\ldots$

        $f_{m}(\mathbf{x}) = (1 + g(\mathbf{x}_m)) \sin(0.5x_{1}^\alpha\pi )$

        Where $m$ is the number of objectives and $\mathbf{x}_m$
        is a vector of the remaining attributes $[x_m~\ldots~x_n]$
        of the individual in $n > m$ dimensions.
    """
    xm = individual[count - 1 :]
    gval = sum((xi - 0.5) ** 2 for xi in xm)
    return _dtlz_helper_1(individual, count, gval, alpha)


def _dtlz_helper_1(
    individual: Individual, count: int, gval: float, alpha: float = 1.0
) -> list[float]:
    """Build DTLZ2-style cosine/sine objectives.

    Args:
        individual: Decision vector.
        count: Number of objectives.
        gval: Distance-to-front term ``g``.
        alpha: Exponent on the angular mapping.

    Returns:
        Objective values of length ``count``.
    """

    def fn(m: int) -> float:
        vals_ = [cos(0.5 * xi**alpha * pi) for xi in xc[:m]]
        rdc = reduce(mul, vals_, 1)
        _sin = sin(0.5 * xc[m] ** alpha * pi)
        return float((1 + gval) * rdc * _sin)

    xc = individual[: count - 1]
    vals = (cos(0.5 * xi**alpha * pi) for xi in xc)
    fit = [(1 + gval) * reduce(mul, vals, 1)]
    vals = [fn(m) for m in range(count - 2, -1, -1)]
    fit.extend(vals)
    return [float(value) for value in fit]
