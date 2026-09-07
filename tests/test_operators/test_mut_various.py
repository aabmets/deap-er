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


def test_bounded_operators_accept_numpy_float32_bounds():
    low: Any = numpy.float32(0.0)
    up: Any = numpy.float32(1.0)
    tools.rng.seed(2)
    (mutant,) = tools.mut_gaussian_bounded([0.5, 0.5], 0.0, 1.0, low, up, 1.0)
    assert all(0.0 <= gene <= 1.0 for gene in mutant)
    tools.rng.seed(3)
    (poly,) = tools.mut_polynomial_bounded([0.5], 20.0, low, up, 1.0)
    assert 0.0 <= poly[0] <= 1.0
    left: Any = [0.2, 0.3]
    right: Any = [0.8, 0.7]
    child1, child2 = tools.cx_blend_bounded(left, right, 0.5, low, up)
    assert all(0.0 <= gene <= 1.0 for gene in child1)
    assert all(0.0 <= gene <= 1.0 for gene in child2)


def test_mut_heterogeneous_applies_one_callable_per_gene():
    tools.rng.seed(6)
    individual: Any = [1, 2, 3]
    mutators = [lambda x: x + 1, lambda x: x * 2, lambda x: 0]
    (mutant,) = tools.mut_heterogeneous(individual, mutators, 1.0)
    assert mutant == [2, 4, 0]
    short: Any = [1, 2]
    with pytest.raises(ValueError, match="same length"):
        tools.mut_heterogeneous(short, [lambda x: x], 1.0)


def test_mut_gaussian_bounded_matches_gaussian_then_clamp():
    genes = [0.2, 0.5, 0.8]
    tools.rng.seed(7)
    unbounded: Any = list(genes)
    (raw,) = tools.mut_gaussian(unbounded, 0.0, 1.0, 0.5)
    expected = [min(max(gene, 0.0), 1.0) for gene in raw]

    tools.rng.seed(7)
    bounded: Any = list(genes)
    (mutant,) = tools.mut_gaussian_bounded(bounded, 0.0, 1.0, 0.0, 1.0, 0.5)
    assert mutant is bounded
    assert mutant == expected


def test_mut_gaussian_bounded_matches_per_gene_bounds():
    genes = [0.4, -0.2, 1.2]
    low = [0.0, -1.0, 0.5]
    up = [1.0, 0.0, 2.0]
    tools.rng.seed(8)
    unbounded: Any = list(genes)
    (raw,) = tools.mut_gaussian(unbounded, [0.0, 0.1, -0.1], [0.5, 1.0, 0.25], 0.8)
    expected = [min(max(gene, xl), xu) for gene, xl, xu in zip(raw, low, up, strict=True)]

    tools.rng.seed(8)
    bounded: Any = list(genes)
    (mutant,) = tools.mut_gaussian_bounded(
        bounded, [0.0, 0.1, -0.1], [0.5, 1.0, 0.25], low, up, 0.8
    )
    assert mutant == expected


def test_mut_gaussian_bounded_out_of_box_stays_in_bounds():
    tools.rng.seed(9)
    individual: Any = [10.0, -5.0]
    (mutant,) = tools.mut_gaussian_bounded(individual, 0.0, 1.0, 0.0, 1.0, 1.0)
    assert mutant is individual
    assert all(0.0 <= gene <= 1.0 for gene in mutant)


def test_mut_gaussian_bounded_zero_prob_is_noop():
    genes = [10.0, -5.0]
    individual: Any = list(genes)
    (mutant,) = tools.mut_gaussian_bounded(individual, 0.0, 1.0, 0.0, 1.0, 0.0)
    assert mutant == genes


def test_mut_gaussian_bounded_skips_empty_interval():
    individual: Any = [0.3, 0.7]
    (mutant,) = tools.mut_gaussian_bounded(individual, 0.0, 1.0, 1.0, 0.0, 1.0)
    assert mutant == [0.3, 0.7]


def test_mut_gaussian_bounded_skips_empty_interval_per_gene():
    genes = [0.3, 0.7]
    low = [1.0, 0.0]
    up = [0.0, 1.0]
    tools.rng.seed(12)
    expected = list(genes)
    for i, xl, xu in zip(range(len(genes)), low, up, strict=True):
        if tools.rng.random() < 1.0:
            if xu <= xl:
                continue
            expected[i] = min(max(expected[i] + tools.rng.gauss(0.0, 1.0), xl), xu)

    tools.rng.seed(12)
    individual: Any = list(genes)
    (mutant,) = tools.mut_gaussian_bounded(individual, 0.0, 1.0, low, up, 1.0)
    assert mutant[0] == genes[0]
    assert mutant == expected
    assert 0.0 <= mutant[1] <= 1.0


def test_mut_gaussian_bounded_unselected_out_of_box_stays():
    genes = [10.0, -5.0]
    mut_prob = 0.5
    tools.rng.seed(0)
    draws = [tools.rng.random() for _ in genes]
    assert draws[0] >= mut_prob
    assert draws[1] < mut_prob

    tools.rng.seed(0)
    expected = list(genes)
    for i, gene in enumerate(expected):
        if tools.rng.random() < mut_prob:
            expected[i] = min(max(gene + tools.rng.gauss(0.0, 1.0), 0.0), 1.0)

    tools.rng.seed(0)
    individual: Any = list(genes)
    (mutant,) = tools.mut_gaussian_bounded(individual, 0.0, 1.0, 0.0, 1.0, mut_prob)
    assert mutant == expected
    assert mutant[0] == genes[0]
    assert 0.0 <= mutant[1] <= 1.0


def test_mut_gaussian_bounded_rejects_short_parameter_sequences():
    individual: Any = [0.0, 1.0]
    with pytest.raises(ValueError, match="at least the size"):
        tools.mut_gaussian_bounded(individual, mu=[0.0], sigma=0.1, low=0.0, up=1.0, mut_prob=1.0)
    with pytest.raises(ValueError, match="at least the size"):
        tools.mut_gaussian_bounded(individual, 0.0, sigma=[0.1], low=0.0, up=1.0, mut_prob=1.0)
    with pytest.raises(ValueError, match="at least the size"):
        tools.mut_gaussian_bounded(individual, 0.0, 0.1, low=[0.0], up=1.0, mut_prob=1.0)
    with pytest.raises(ValueError, match="at least the size"):
        tools.mut_gaussian_bounded(individual, 0.0, 0.1, 0.0, up=[1.0], mut_prob=1.0)


def test_mut_gaussian_bounded_accepts_numpy_integer_bounds():
    individual: Any = [10.0, -5.0]
    low: Any = numpy.int64(0)
    up: Any = numpy.int64(1)
    tools.rng.seed(1)
    (mutant,) = tools.mut_gaussian_bounded(individual, 0.0, 1.0, low, up, 1.0)
    assert all(0.0 <= gene <= 1.0 for gene in mutant)
