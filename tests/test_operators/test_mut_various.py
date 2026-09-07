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

import numpy
import pytest
from deap_er import tools


class _ESList(list[Any]):
    def __init__(self, genes: list[float], strategy: list[float]) -> None:
        super().__init__(genes)
        self.strategy = list(strategy)


def test_mut_shuffle_indexes_length_one_is_noop():
    individual: Any = [0]
    assert tools.mut_shuffle_indexes(individual, 1.0) == ([0],)


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


def test_flip_bit_matches_per_gene_random_stream():
    genes = [0, 1, 0, 1, 1, 0, 0, 1]
    tools.rng.seed(11)
    draws = [tools.rng.random() for _ in range(len(genes))]
    expected = [
        type(gene)(not gene) if draw < 0.35 else gene
        for gene, draw in zip(genes, draws, strict=True)
    ]
    tools.rng.seed(11)
    individual: Any = list(genes)
    (mutant,) = tools.mut_flip_bit(individual, 0.35)
    assert mutant == expected


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


def test_polynomial_bounded_out_of_box_stays_finite():
    tools.rng.seed(5)
    individual: Any = [10.0, -5.0]
    (mutant,) = tools.mut_polynomial_bounded(individual, 2.0, 0.0, 1.0, 1.0)
    assert all(0.0 <= gene <= 1.0 for gene in mutant)
    assert all(gene == gene for gene in mutant)


def test_mut_uniform_int_accepts_numpy_integer_bounds():
    individual: Any = [0, 0, 0]
    low: Any = numpy.int64(0)
    up: Any = numpy.int64(3)
    tools.rng.seed(1)
    (mutant,) = tools.mut_uniform_int(individual, low, up, 1.0)
    assert all(0 <= gene <= 3 for gene in mutant)


def test_mut_heterogeneous_applies_one_callable_per_gene():
    tools.rng.seed(6)
    individual: Any = [1, 2, 3]
    mutators = [lambda x: x + 1, lambda x: x * 2, lambda x: 0]
    (mutant,) = tools.mut_heterogeneous(individual, mutators, 1.0)
    assert mutant == [2, 4, 0]
    short: Any = [1, 2]
    with pytest.raises(ValueError, match="same length"):
        tools.mut_heterogeneous(short, [lambda x: x], 1.0)
