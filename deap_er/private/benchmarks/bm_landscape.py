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

from math import sin, sqrt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = ["bm_schaffer", "bm_schwefel", "bm_himmelblau"]


def bm_schaffer(individual: Individual) -> tuple[float]:
    r"""Schaffer test objective function.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness value of the individual.

    ??? note "Equations"

        | | |
        |---|---|
        | Type | minimization |
        | Range | $x_i \in [-100, 100]$ |
        | Global optima | $x_i = 0, \forall i \in \lbrace 1 \ldots N\rbrace$, $f(\mathbf{x}) = 0$ |
        | Function | see below |

        $$
        f(\mathbf{x}) = \sum_{i=1}^{N-1}
        (x_i^2+x_{i+1}^2)^{0.25} \cdot \left[
        \sin^2(50\cdot(x_i^2+x_{i+1}^2)^{0.10})
        + 1.0 \right]
        $$
    """
    results = []
    for x, x1 in zip(individual[:-1], individual[1:], strict=False):
        var_1 = (x**2 + x1**2) ** 0.25
        var_2 = sin(50 * (x**2 + x1**2) ** 0.1) ** 2 + 1.0
        results.append(var_1 * var_2)
    result = sum(results)
    return (float(result),)


def bm_schwefel(individual: Individual) -> tuple[float]:
    r"""Schwefel test objective function.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness value of the individual.

    ??? note "Equations"

        | | |
        |---|---|
        | Type | minimization |
        | Range | $x_i \in [-500, 500]$ |
        | Global optima | see below |
        | Function | see below |

        $x_i = 420.96874636$, $\forall i \in \lbrace 1 \ldots N\rbrace$,
        $f(\mathbf{x}) = 0$

        $$
        f(\mathbf{x}) = 418.9828872724339\cdot N -
        \sum_{i=1}^N\,x_i\sin\left(\sqrt{|x_i|}\right)
        $$
    """
    len_ind = len(individual)
    values = sum(x * sin(sqrt(abs(x))) for x in individual)
    result = 418.9828872724339 * len_ind - values
    return (float(result),)


def bm_himmelblau(individual: Individual) -> tuple[float]:
    r"""The Himmelblau function has four minima in $[-6, 6]^2$.

    Args:
        individual: Individual to evaluate.

    Returns:
        Fitness value of the individual.

    ??? note "Equations"

        | | |
        |---|---|
        | Type | minimization |
        | Range | $x_i \in [-6, 6]$ |
        | Global optima | see below |
        | Function | see below |

        $\mathbf{x}_1 = (3.0, 2.0)$, $f(\mathbf{x}_1) = 0$

        $\mathbf{x}_2 = (-2.805118, 3.131312)$, $f(\mathbf{x}_2) = 0$

        $\mathbf{x}_3 = (-3.779310, -3.283186)$, $f(\mathbf{x}_3) = 0$

        $\mathbf{x}_4 = (3.584428, -1.848126)$, $f(\mathbf{x}_4) = 0$

        $$
        f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 -7)^2
        $$
    """
    var_1 = (individual[0] * individual[0] + individual[1] - 11) ** 2
    var_2 = (individual[0] + individual[1] * individual[1] - 7) ** 2
    result = var_1 + var_2
    return (float(result),)
