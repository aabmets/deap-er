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
from math import cos, exp, pi, sin, sqrt
from operator import mul

from deap_er.base.typedefs import Individual

__all__ = [
    "bm_kursawe",
    "bm_schaffer_mo",
    "bm_fonseca",
    "bm_poloni",
    "bm_dent",
    "bm_zdt_1",
    "bm_zdt_2",
    "bm_zdt_3",
    "bm_zdt_4",
    "bm_zdt_6",
    "bm_dtlz_1",
    "bm_dtlz_2",
    "bm_dtlz_3",
    "bm_dtlz_4",
    "bm_dtlz_5",
    "bm_dtlz_6",
    "bm_dtlz_7",
]


def _zdt_g(individual: Individual) -> float:
    r"""Return the ZDT1, ZDT2, and ZDT3 distance term.

    ZDT4 and ZDT6 define their own distance terms and do not use this.

    Args:
        individual: Individual to evaluate.

    Returns:
        The value of :math:`g(\mathbf{x})` for the individual.
    """
    return float(1.0 + 9.0 * sum(individual[1:]) / (len(individual) - 1))


def bm_kursawe(individual: Individual) -> tuple[float, float]:
    r"""Kursawe multi-objective function.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness values of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`f_{1}(\mathbf{x}) = \sum_{i=1}^{N-1} -10 e^{-0.2 \sqrt{x_i^2 + x_{i+1}^2} }`

       :math:`f_{2}(\mathbf{x}) = \sum_{i=1}^{N} |x_i|^{0.8} + 5 \sin(x_i^3)`

       Returns :math:`f_{1}(\mathbf{x})` and :math:`f_{2}(\mathbf{x})`.
    """

    def fn(x: float, y: float) -> float:
        return -10 * exp(-0.2 * sqrt(x * x + y * y))

    f1 = sum(fn(x, y) for x, y in zip(individual[:-1], individual[1:], strict=False))
    f2 = sum(abs(x) ** 0.8 + 5 * sin(x * x * x) for x in individual)
    return float(f1), float(f2)


def bm_schaffer_mo(individual: Individual) -> tuple[float, float]:
    r"""Schaffer's multi-objective function on a one-attribute **individual**.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness values of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`f_{1}(\mathbf{x}) = x_1^2`

       :math:`f_{2}(\mathbf{x}) = (x_1-2)^2`

       Returns :math:`f_{1}(\mathbf{x})` and :math:`f_{2}(\mathbf{x})`.
    """
    f1 = individual[0] ** 2
    f2 = (individual[0] - 2) ** 2
    return float(f1), float(f2)


def bm_fonseca(individual: Individual) -> tuple[float, float]:
    r"""Fonseca and Fleming's multiobjective function.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness values of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`f_{1}(\mathbf{x}) = 1 - e^{-\sum_{i=1}^{3}(x_i - \frac{1}{\sqrt{3}})^2}`

       :math:`f_{2}(\mathbf{x}) = 1 - e^{-\sum_{i=1}^{3}(x_i + \frac{1}{\sqrt{3}})^2}`

       Returns :math:`f_{1}(\mathbf{x})` and :math:`f_{2}(\mathbf{x})`.
    """
    f1 = 1 - exp(-sum((xi - 1 / sqrt(3)) ** 2 for xi in individual[:3]))
    f2 = 1 - exp(-sum((xi + 1 / sqrt(3)) ** 2 for xi in individual[:3]))
    return float(f1), float(f2)


def bm_poloni(individual: Individual) -> tuple[float, float]:
    r"""Poloni's multiobjective function on a two-attribute **individual**.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness values of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`A_1 = 0.5 \sin (1) - 2 \cos (1) + \sin (2) - 1.5 \cos (2)`

       :math:`A_2 = 1.5 \sin (1) - \cos (1) + 2 \sin (2) - 0.5 \cos (2)`

       :math:`B_1 = 0.5 \sin (x_1) - 2 \cos (x_1) + \sin (x_2) - 1.5 \cos (x_2)`

       :math:`B_2 = 1.5 \sin (x_1) - cos(x_1) + 2 \sin (x_2) - 0.5 \cos (x_2)`

       :math:`f_{1}(\mathbf{x}) = 1 + (A_1 - B_1)^2 + (A_2 - B_2)^2`

       :math:`f_{2}(\mathbf{x}) = (x_1 + 3)^2 + (x_2 + 1)^2`

       Returns :math:`f_{1}(\mathbf{x})` and :math:`f_{2}(\mathbf{x})`.
    """
    x_1 = individual[0]
    x_2 = individual[1]
    a_1 = 0.5 * sin(1) - 2 * cos(1) + sin(2) - 1.5 * cos(2)
    a_2 = 1.5 * sin(1) - cos(1) + 2 * sin(2) - 0.5 * cos(2)
    b_1 = 0.5 * sin(x_1) - 2 * cos(x_1) + sin(x_2) - 1.5 * cos(x_2)
    b_2 = 1.5 * sin(x_1) - cos(x_1) + 2 * sin(x_2) - 0.5 * cos(x_2)
    f1 = 1 + (a_1 - b_1) ** 2 + (a_2 - b_2) ** 2
    f2 = (x_1 + 3) ** 2 + (x_2 + 1) ** 2
    return float(f1), float(f2)


def bm_dent(individual: Individual, dent_size: float = 0.85) -> tuple[float, float]:
    r"""Evaluate a two-objective problem with a dent.

    The individual must have two attributes in ``[-1.5, 1.5]``.

    Args:
        individual: Individual to evaluate.
        dent_size: Size of the dent.

    Returns:
        Fitness values of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`f_{1}(\mathbf{x}) = \text{ ?}`

       :math:`f_{2}(\mathbf{x}) = \text{ ?}`

       Returns :math:`f_{1}(\mathbf{x})` and :math:`f_{2}(\mathbf{x})`.
    """
    d = dent_size * exp(-((individual[0] - individual[1]) ** 2))
    f1 = (
        0.5
        * (
            sqrt(1 + (individual[0] + individual[1]) ** 2)
            + sqrt(1 + (individual[0] - individual[1]) ** 2)
            + individual[0]
            - individual[1]
        )
        + d
    )
    f2 = (
        0.5
        * (
            sqrt(1 + (individual[0] + individual[1]) ** 2)
            + sqrt(1 + (individual[0] - individual[1]) ** 2)
            - individual[0]
            + individual[1]
        )
        + d
    )
    return float(f1), float(f2)


def bm_zdt_1(individual: Individual) -> tuple[float, float]:
    r"""ZDT1 multi-objective function.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness values of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\mathbf{x}) = 1 + \frac{9}{n-1}\sum_{i=2}^n x_i`

       :math:`f_{1}(\mathbf{x}) = x_1`

       :math:`f_{2}(\mathbf{x}) = g(\mathbf{x})\left[1 -             \sqrt{\frac{x_1}{g(\mathbf{x})}}\right]`

       Returns :math:`f_{1}(\mathbf{x})` and :math:`f_{2}(\mathbf{x})`.
    """
    g = _zdt_g(individual)
    f1 = individual[0]
    f2 = g * (1 - sqrt(f1 / g))
    return float(f1), float(f2)


def bm_zdt_2(individual: Individual) -> tuple[float, float]:
    r"""ZDT2 multi-objective function.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness values of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\mathbf{x}) = 1 + \frac{9}{n-1}\sum_{i=2}^n x_i`

       :math:`f_{1}(\mathbf{x}) = x_1`

       :math:`f_{2}(\mathbf{x}) = g(\mathbf{x})\left[1 -             \left(\frac{x_1}{g(\mathbf{x})}\right)^2\right]`

       Returns :math:`f_{1}(\mathbf{x})` and :math:`f_{2}(\mathbf{x})`.
    """
    g = _zdt_g(individual)
    f1 = individual[0]
    f2 = g * (1 - (f1 / g) ** 2)
    return float(f1), float(f2)


def bm_zdt_3(individual: Individual) -> tuple[float, float]:
    r"""ZDT3 multi-objective function.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness values of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\mathbf{x}) = 1 + \frac{9}{n-1}\sum_{i=2}^n x_i`

       :math:`f_{1}(\mathbf{x}) = x_1`

       :math:`f_{2}(\mathbf{x}) = g(\mathbf{x})\left[1 -             \sqrt{\frac{x_1}{g(\mathbf{x})}} - \frac{x_1}{g(\mathbf{x})}             \sin(10\pi x_1)\right]`

       Returns :math:`f_{1}(\mathbf{x})` and :math:`f_{2}(\mathbf{x})`.
    """
    g = _zdt_g(individual)
    f1 = individual[0]
    f2 = g * (1 - sqrt(f1 / g) - f1 / g * sin(10 * pi * f1))
    return float(f1), float(f2)


def bm_zdt_4(individual: Individual) -> tuple[float, float]:
    r"""ZDT4 multi-objective function.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness values of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\mathbf{x}) = 1 + 10(n-1) + \sum_{i=2}^n             \left[ x_i^2 - 10\cos(4\pi x_i) \right]`

       :math:`f_{1}(\mathbf{x}) = x_1`

       :math:`f_{2}(\mathbf{x}) = g(\mathbf{x}) \left[ 1 -             \sqrt{ \frac{x_1}{g(\mathbf{x})}} \right]`

       Returns :math:`f_{1}(\mathbf{x})` and :math:`f_{2}(\mathbf{x})`.
    """
    var = sum(xi**2 - 10 * cos(4 * pi * xi) for xi in individual[1:])
    g = 1 + 10 * (len(individual) - 1) + var
    f1 = individual[0]
    f2 = g * (1 - sqrt(f1 / g))
    return float(f1), float(f2)


def bm_zdt_6(individual: Individual) -> tuple[float, float]:
    r"""ZDT6 multi-objective function.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness values of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\mathbf{x}) = 1 + 9 \left[ \left(\sum_{i=2}^n             x_i\right)/(n-1) \right]^{0.25}`

       :math:`f_{1}(\mathbf{x}) = 1 - \exp(-4x_1)\sin^6(6\pi x_1)`

       :math:`f_{2}(\mathbf{x}) = g(\mathbf{x}) \left[1 - \left(             \frac{f_{1}(\mathbf{x})}{g(\mathbf{x})}\right)^2 \right]`

       Returns :math:`f_{1}(\mathbf{x})` and :math:`f_{2}(\mathbf{x})`.
    """
    g = 1 + 9 * (sum(individual[1:]) / (len(individual) - 1)) ** 0.25
    f1 = 1 - exp(-4 * individual[0]) * sin(6 * pi * individual[0]) ** 6
    f2 = g * (1 - (f1 / g) ** 2)
    return float(f1), float(f2)


def bm_dtlz_1(individual: Individual, count: int) -> list[float]:
    r"""Evaluate the DTLZ1 multi-objective function.

    Returns a list of size ``count``. The individual must have at
    least ``count`` elements.

    Args:
        individual: Individual to evaluate.
        count: Number of objectives.

    Returns:
        Fitness values of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\mathbf{x}_m) = 100\left(|\mathbf{x}_m| + \sum_{x_i             \in \mathbf{x}_m}\left((x_i - 0.5)^2 -             \cos(20\pi(x_i - 0.5))\right)\right)`

       :math:`f_{1}(\mathbf{x}) = \frac{1}{2} (1 +             g(\mathbf{x}_m)) \prod_{i=1}^{m-1}x_i`

       :math:`f_{2}(\mathbf{x}) = \frac{1}{2} (1 + g(\mathbf{x}_m))             (1-x_{m-1}) \prod_{i=1}^{m-2}x_i`

       :math:`f_{m-1}(\mathbf{x}) = \frac{1}{2} (1 +             g(\mathbf{x}_m)) (1 - x_2) x_1`

       :math:`\ldots`

       :math:`f_{m}(\mathbf{x}) = \frac{1}{2}             (1 - x_1)(1 + g(\mathbf{x}_m))`


       Where :math:`m` is the number of objectives and :math:`\mathbf{x}_m`
       is a vector of the remaining attributes :math:`[x_m~\ldots~x_n]`
       of the individual in :math:`n > m` dimensions.
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

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\mathbf{x}_m) = \sum_{x_i \in             \mathbf{x}_m} (x_i - 0.5)^2`

       :math:`f_{1}(\mathbf{x}) = (1 + g(\mathbf{x}_m))             \prod_{i=1}^{m-1} \cos(0.5x_i\pi)`

       :math:`f_{2}(\mathbf{x}) = (1 + g(\mathbf{x}_m))             \sin(0.5x_{m-1}\pi ) \prod_{i=1}^{m-2} \cos(0.5x_i\pi)`

       :math:`\ldots`

       :math:`f_{m}(\mathbf{x}) = (1 +             g(\mathbf{x}_m)) \sin(0.5x_{1}\pi )`


       Where :math:`m` is the number of objectives and :math:`\mathbf{x}_m`
       is a vector of the remaining attributes :math:`[x_m~\ldots~x_n]`
       of the individual in :math:`n > m` dimensions.
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

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\mathbf{x}_m) = 100\left(|\mathbf{x}_m| +             \sum_{x_i \in \mathbf{x}_m}\left((x_i - 0.5)^2 -             \cos(20\pi(x_i - 0.5))\right)\right)`

       :math:`f_{1}(\mathbf{x}) = (1 + g(\mathbf{x}_m))             \prod_{i=1}^{m-1} \cos(0.5x_i\pi)`

       :math:`f_{2}(\mathbf{x}) = (1 + g(\mathbf{x}_m))             \sin(0.5x_{m-1}\pi ) \prod_{i=1}^{m-2} \cos(0.5x_i\pi)`

       :math:`\ldots`

       :math:`f_{m}(\mathbf{x}) = (1 + g(\mathbf{x}_m))             \sin(0.5x_{1}\pi )`


       Where :math:`m` is the number of objectives and :math:`\mathbf{x}_m`
       is a vector of the remaining attributes :math:`[x_m~\ldots~x_n]`
       of the individual in :math:`n > m` dimensions.
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

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\mathbf{x}_m) = \sum_{x_i \in             \mathbf{x}_m} (x_i - 0.5)^2`

       :math:`f_{1}(\mathbf{x}) = (1 + g(\mathbf{x}_m))             \prod_{i=1}^{m-1} \cos(0.5x_i^\alpha\pi)`

       :math:`f_{2}(\mathbf{x}) = (1 + g(\mathbf{x}_m))             \sin(0.5x_{m-1}^\alpha\pi ) \prod_{i=1}^{m-2}             \cos(0.5x_i^\alpha\pi)`

       :math:`\ldots`

       :math:`f_{m}(\mathbf{x}) = (1 + g(\mathbf{x}_m))             \sin(0.5x_{1}^\alpha\pi )`


       Where :math:`m` is the number of objectives and :math:`\mathbf{x}_m`
       is a vector of the remaining attributes :math:`[x_m~\ldots~x_n]`
       of the individual in :math:`n > m` dimensions.
    """
    xm = individual[count - 1 :]
    gval = sum((xi - 0.5) ** 2 for xi in xm)
    return _dtlz_helper_1(individual, count, gval, alpha)


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
