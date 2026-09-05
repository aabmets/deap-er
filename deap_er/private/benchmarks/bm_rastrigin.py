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

from math import cos, pi
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = ["bm_rastrigin", "bm_rastrigin_scaled", "bm_rastrigin_skewed", "bm_shekel"]


def bm_rastrigin(individual: Individual) -> tuple[float]:
    r"""Rastrigin test objective function.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness value of the individual.

    ??? note "Equations"

        | | |
        |---|---|
        | Type | minimization |
        | Range | $x_i \in [-5.12, 5.12]$ |
        | Global optima | $x_i = 0, \forall i \in \lbrace 1 \ldots N\rbrace$, $f(\mathbf{x}) = 0$ |
        | Function | $f(\mathbf{x}) = 10N + \sum_{i=1}^N x_i^2 - 10 \cos(2\pi x_i)$ |
    """
    values = [gene * gene - 10 * cos(2 * pi * gene) for gene in individual]
    result = 10 * len(individual) + sum(values)
    return (float(result),)


def bm_rastrigin_scaled(individual: Individual) -> tuple[float]:
    r"""Scaled Rastrigin test objective function.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness value of the individual.

    ??? note "Equations"

        | | |
        |---|---|
        | Type | minimization |
        | Range | $x_i \in [-5.12, 5.12]$ |
        | Global optima | $x_i = 0, \forall i \in \lbrace 1 \ldots N\rbrace$, $f(\mathbf{x}) = 0$ |
        | Function | see below |

        $$
        f(\mathbf{x}) = 10N + \sum_{i=1}^N
        \left(10^{\left(\frac{i-1}{N-1}\right)}
        x_i \right)^2 - 10\cos\left(2\pi
        10^{\left(\frac{i-1}{N-1}\right)} x_i
        \right)
        $$
    """
    results = []
    len_ind = len(individual)
    for i, x in enumerate(individual):
        var_1 = (10 ** (i / (len_ind - 1)) * x) ** 2
        var_2 = 10 * cos(2 * pi * 10 ** (i / (len_ind - 1)) * x)
        results.append(var_1 - var_2)
    result = 10 * len_ind + sum(results)
    return (float(result),)


def bm_rastrigin_skewed(individual: Individual) -> tuple[float]:
    r"""Skewed Rastrigin test objective function.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness value of the individual.

    ??? note "Equations"

        | | |
        |---|---|
        | Type | minimization |
        | Range | $x_i \in [-5.12, 5.12]$ |
        | Global optima | $x_i = 0, \forall i \in \lbrace 1 \ldots N\rbrace$, $f(\mathbf{x}) = 0$ |
        | Function | see below |

        $f(\mathbf{x}) = 10N + \sum_{i=1}^N \left(y_i^2 - 10 \cos(2\pi x_i)\right)$

        $\text{where } y_i = 10\cdot x_i \text{ if } x_i > 0 \text{, else } x_i$
    """
    results = []
    len_ind = len(individual)
    for x in individual:
        var_1 = (10 * x if x > 0 else x) ** 2
        var_2 = 10 * cos(2 * pi * (10 * x if x > 0 else x))
        results.append(var_1 - var_2)
    result = 10 * len_ind + sum(results)
    return (float(result),)


def bm_shekel(individual: Individual, matrix: numpy.ndarray, vector: numpy.ndarray) -> tuple[float]:
    r"""Evaluate the Shekel multimodal function.

    The number of maxima is the length of ``matrix`` and ``vector``.

    Args:
        individual: Individual to evaluate.
        matrix: Matrix of size $M\times N$,
            where $M$ is the number of maxima and
            $N$ is the number of dimensions.
        vector: Vector of size $M\times 1$,
            where $M$ is the number of maxima.

    Returns:
        Fitness value of the individual.

    ??? note "Equations"

        | | |
        |---|---|
        | Type | maximization |
        | Range | None |
        | Global optima | None |
        | Function | see below |

        $$
        f(\mathbf{x}) = \sum_{i = 1}^{M}
        \frac{1}{c_{i} + \sum_{j = 1}^{N}
        (x_{j} - a_{ij})^2 }
        $$
    """
    results = []
    for i in range(len(vector)):
        values = []
        for j, g in enumerate(matrix[i]):
            val = (individual[j] - g) ** 2
            values.append(val)
        result = 1 / (vector[i] + sum(values))
        results.append(result)
    result = sum(results)
    return (float(result),)
