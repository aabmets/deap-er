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
from math import cos, e, exp, pi, sin, sqrt
from operator import mul

from deap_er.base.typedefs import Individual

__all__: list[str] = []


def bm_h1(individual: Individual) -> tuple[float]:
    r"""Simple two-dimensional function containing several local maxima.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness value of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - maximization
          * - Range
            - :math:`x_i \in [-100, 100]`
          * - Global optima
            - :math:`\mathbf{x} = (8.6998, 6.7665)`, :math:`f(\mathbf{x}) = 2`

          * - Function
            - :math:`f(\mathbf{x}) = \frac{\sin(x_1 - \frac{x_2}{8})^2 +                \sin(x_2 + \frac{x_1}{8})^2}{\sqrt{(x_1 - 8.6998)^2 +                (x_2 - 6.7665)^2} + 1}`
    """

    def compute_num() -> float:
        var_1 = sin(individual[0] - individual[1] / 8) ** 2
        var_2 = sin(individual[1] + individual[0] / 8) ** 2
        return float(var_1 + var_2)

    def compute_denum() -> float:
        var_1 = (individual[0] - 8.6998) ** 2
        var_2 = (individual[1] - 6.7665) ** 2
        return float((var_1 + var_2) ** 0.5 + 1)

    result = compute_num() / compute_denum()
    return (float(result),)


def bm_ackley(individual: Individual) -> tuple[float]:
    r"""Ackley test objective function.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness value of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - :math:`x_i \in [-15, 30]`
          * - Global optima
            - :math:`x_i = 0, \forall i \in \lbrace 1 \ldots                N\rbrace`, :math:`f(\mathbf{x}) = 0`
          * - Function
            - :math:`f(\mathbf{x}) = 20 - 20\exp\left(-0.2                \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}                \right) + e - \exp\left(\frac{1}{N}                \sum_{i=1}^N \cos(2\pi x_i) \right)`
    """
    len_ind = len(individual)
    exp_1 = exp(-0.2 * sqrt(1 / len_ind * sum(x**2 for x in individual)))
    exp_2 = exp(1 / len_ind * sum(cos(2 * pi * x) for x in individual))
    result = 20 - 20 * exp_1 + e - exp_2
    return (float(result),)


def bm_bohachevsky(individual: Individual) -> tuple[float]:
    r"""Bohachevsky test objective function.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness value of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - :math:`x_i \in [-100, 100]`
          * - Global optima
            - :math:`x_i = 0, \forall i \in \lbrace 1 \ldots                N\rbrace`, :math:`f(\mathbf{x}) = 0`
          * - Function
            - :math:`f(\mathbf{x}) = \sum_{i=1}^{N-1}(x_i^2 +                2x_{i+1}^2 - 0.3\cos(3\pi x_i) - 0.4\cos(4                \pi x_{i+1}) + 0.7)`
    """
    results = []
    for x, x1 in zip(individual[:-1], individual[1:], strict=False):
        c1 = cos(3 * pi * x)
        c2 = cos(4 * pi * x1)
        res = x**2 + 2 * x1**2 - 0.3 * c1 - 0.4 * c2 + 0.7
        results.append(res)
    result = sum(results)
    return (float(result),)


def bm_griewank(individual: Individual) -> tuple[float]:
    r"""Griewank test objective function.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness value of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - :math:`x_i \in [-600, 600]`
          * - Global optima
            - :math:`x_i = 0, \forall i \in \lbrace 1 \ldots                N\rbrace`, :math:`f(\mathbf{x}) = 0`
          * - Function
            - :math:`f(\mathbf{x}) = \frac{1}{4000}\sum_{i=1}^N                \,x_i^2 - \prod_{i=1}^N\cos\left(                \frac{x_i}{\sqrt{i}}\right) + 1`
    """
    values = [cos(x / sqrt(i + 1.0)) for i, x in enumerate(individual)]
    exp_sum = sum(x**2 for x in individual)
    result = 1 / 4000 * exp_sum - reduce(mul, values, 1) + 1
    return (float(result),)
