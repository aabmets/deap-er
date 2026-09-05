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

from math import cos, exp, pi, sin, sqrt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = ["bm_zdt_1", "bm_zdt_2", "bm_zdt_3", "bm_zdt_4", "bm_zdt_6"]


def _zdt_g(individual: Individual) -> float:
    r"""Return the ZDT1, ZDT2, and ZDT3 distance term.

    ZDT4 and ZDT6 define their own distance terms and do not use this.

    Args:
        individual: Individual to evaluate.

    Returns:
        The value of :math:`g(\mathbf{x})` for the individual.
    """
    return float(1.0 + 9.0 * sum(individual[1:]) / (len(individual) - 1))


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
