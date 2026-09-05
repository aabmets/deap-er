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

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual, Mutant
from deap_er.private.various.rng import rng

__all__: list[str] = ["mut_heterogeneous"]


def mut_heterogeneous(
    individual: Individual,
    mutators: Sequence[Callable[[Any], Any]],
    mut_prob: float,
) -> Mutant:
    """Mutate each gene with its own callable.

    The individual is modified in place. ``mutators[i]`` receives the
    current value of gene ``i`` and must return the replacement.

    Args:
        individual: Individual to mutate.
        mutators: One gene mutator per attribute.
        mut_prob: Probability of mutating each attribute.

    Returns:
        A one-element tuple containing the mutated individual.

    Raises:
        ValueError: If ``mutators`` and ``individual`` have different
            lengths.
    """
    if len(mutators) != len(individual):
        raise ValueError(
            "mutators must have the same length as the individual: "
            f"{len(mutators)} != {len(individual)}"
        )
    for i, mutate_gene in enumerate(mutators):
        if rng.random() < mut_prob:
            individual[i] = mutate_gene(individual[i])
    return (individual,)
