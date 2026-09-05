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
from deap_er.base.typedefs import Individual
from deap_er.rng import rng

__all__: list[str] = []


def bm_rand(*_) -> tuple[float]:
    r"""Random test objective function. Unused extra arguments are ignored.

    Returns:
        A uniformly random number in ``[0, 1)``.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization or maximization
          * - Range
            - none
          * - Global optima
            - none
          * - Function
            - :math:`f(\mathbf{x}) = \text{random}(0,1)`
    """
    result = rng.random()
    return (float(result),)


def bm_plane(individual: Individual) -> tuple[float]:
    r"""Plane test objective function.

    Args:
        individual: Individual to evaluate.

    Returns:
        The first attribute of the individual.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - none
          * - Global optima
            - :math:`x_i = 0, \forall i \in \lbrace 1 \ldots                N\rbrace`, :math:`f(\mathbf{x}) = 0`
          * - Function
            - :math:`f(\mathbf{x}) = x_0`
    """
    result = individual[0]
    return (float(result),)


def bm_sphere(individual: Individual) -> tuple[float]:
    r"""Sphere test objective function.

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
            - none
          * - Global optima
            - :math:`x_i = 0, \forall i \in \lbrace 1 \ldots                N\rbrace`, :math:`f(\mathbf{x}) = 0`
          * - Function
            - :math:`f(\mathbf{x}) = \sum_{i=1}^Nx_i^2`
    """
    result = sum(gene * gene for gene in individual)
    return (float(result),)


def bm_cigar(individual: Individual) -> tuple[float]:
    r"""Cigar test objective function.

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
            - none
          * - Global optima
            - :math:`x_i = 0, \forall i \in \lbrace 1 \ldots                N\rbrace`, :math:`f(\mathbf{x}) = 0`
          * - Function
            - :math:`f(\mathbf{x}) = x_0^2 + 10^6\sum_{i=1}^N\,x_i^2`
    """
    _sum = sum(gene * gene for gene in individual[1:])
    result = individual[0] ** 2 + 1e6 * _sum
    return (float(result),)


def bm_rosenbrock(individual: Individual) -> tuple[float]:
    r"""Rosenbrock test objective function.

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
            - none
          * - Global optima
            - :math:`x_i = 1, \forall i \in \lbrace 1 \ldots                N\rbrace`, :math:`f(\mathbf{x}) = 0`
          * - Function
            - :math:`f(\mathbf{x}) = \sum_{i=1}^{N-1}                (1-x_i)^2 + 100 (x_{i+1} - x_i^2 )^2`
    """
    results = []
    for x, y in zip(individual[:-1], individual[1:], strict=False):
        results.append(100 * (x * x - y) ** 2 + (1 - x) ** 2)
    result = sum(results)
    return (float(result),)
