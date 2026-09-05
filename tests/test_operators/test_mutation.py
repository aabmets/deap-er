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
from typing import Any

from deap_er import tools


class _ESList(list[Any]):
    def __init__(self, genes: list[float], strategy: list[float]) -> None:
        super().__init__(genes)
        self.strategy = list(strategy)


def test_shuffle_indexes_preserves_genes():
    genes = [0, 1, 2, 3, 4, 5]
    individual: Any = list(genes)
    tools.rng.seed(1)
    (mutant,) = tools.mut_shuffle_indexes(individual, 1.0)
    assert sorted(mutant) == genes
    assert mutant != genes


def test_flip_bit_inverts_boolean_genes():
    individual: Any = [0, 1, 0, 1]
    tools.rng.seed(2)
    (mutant,) = tools.mut_flip_bit(individual, 1.0)
    assert mutant == [1, 0, 1, 0]


def test_es_log_normal_updates_strategy_and_genes():
    individual: Any = _ESList([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    tools.rng.seed(3)
    (mutant,) = tools.mut_es_log_normal(individual, 1.0, 1.0)
    assert mutant is individual
    assert mutant != [0.0, 0.0, 0.0]
    assert all(sigma > 0 for sigma in individual.strategy)


def test_es_log_normal_skips_individuals_without_strategy():
    individual: Any = [1.0, 2.0]
    tools.rng.seed(4)
    (mutant,) = tools.mut_es_log_normal(individual, 1.0, 1.0)
    assert mutant == [1.0, 2.0]
