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
from math import cos, exp, sin, sqrt

from deap_er.base.typedefs import Individual

__all__: list[str] = []


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
