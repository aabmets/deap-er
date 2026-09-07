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
import array
from typing import Any

import pytest
from deap_er import tools


class _Marked(list[Any]):
    def __init__(self, genes: list[float], fitness: tuple[float, ...] = (1.0,)) -> None:
        super().__init__(genes)
        self.fitness = fitness


def _expected_trial(
    parent: list[float],
    a: list[float],
    b: list[float],
    c: list[float],
    scale: float,
    cx_prob: float,
) -> list[float]:
    size = len(parent)
    index = tools.rng.randrange(size)
    draws = tools.rng.take_floats(size)
    trial = list(parent)
    for i, draw in enumerate(draws):
        if i == index or draw < cx_prob:
            trial[i] = a[i] + scale * (b[i] - c[i])
    return trial


def test_mut_de_matches_binomial_formula():
    parent = [0.0, 1.0, 2.0, 3.0]
    a, b, c = [1.0, 1.0, 1.0, 1.0], [4.0, 0.0, 3.0, 2.0], [2.0, 1.0, 0.0, 5.0]
    tools.rng.seed(11)
    expected = _expected_trial(parent, a, b, c, 0.5, 0.4)
    tools.rng.seed(11)
    individual: Any = list(parent)
    (trial,) = tools.mut_de(individual, a, b, c, 0.5, 0.4)
    assert trial is individual
    assert trial == expected


def test_mut_de_cx_prob_zero_writes_one_gene():
    parent = [0.0, 0.0, 0.0, 0.0, 0.0]
    a, b, c = [1.0] * 5, [3.0] * 5, [1.0] * 5
    tools.rng.seed(3)
    individual: Any = list(parent)
    (trial,) = tools.mut_de(individual, a, b, c, 1.0, 0.0)
    changed = [i for i, gene in enumerate(trial) if gene != 0.0]
    assert len(changed) == 1
    assert trial[changed[0]] == 3.0


def test_mut_de_cx_prob_one_writes_all_genes():
    individual: Any = [0.0, 0.0, 0.0]
    (trial,) = tools.mut_de(individual, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 1.0, 1.0], 0.5, 1.0)
    assert trial == [2.5, 4.0, 5.5]


def test_mut_de_leaves_fitness_untouched():
    individual: Any = _Marked([0.0, 1.0], fitness=(9.0,))
    (trial,) = tools.mut_de(individual, [1.0, 1.0], [2.0, 2.0], [0.0, 0.0], 1.0, 1.0)
    assert trial.fitness == (9.0,)


def test_mut_de_accepts_array_array():
    individual: Any = array.array("d", [0.0, 0.0])
    (trial,) = tools.mut_de(
        individual,
        array.array("d", [1.0, 2.0]),
        array.array("d", [3.0, 4.0]),
        array.array("d", [1.0, 1.0]),
        1.0,
        1.0,
    )
    assert list(trial) == [3.0, 5.0]


def test_mut_de_empty_individual_is_noop():
    individual: Any = []
    assert tools.mut_de(individual, [], [], [], 1.0, 1.0) == ([],)


def test_mut_de_rejects_short_donor():
    individual: Any = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="Donors a, b, and c"):
        tools.mut_de(individual, [1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1.0, 1.0)


def test_mut_de_rejects_half_bounds():
    individual: Any = [0.0, 0.0]
    with pytest.raises(ValueError, match="low"):
        tools.mut_de(individual, [1.0, 1.0], [2.0, 2.0], [0.0, 0.0], 1.0, 1.0, low=0.0)


def test_mut_de_clamps_written_genes():
    individual: Any = [10.0, -5.0]
    (trial,) = tools.mut_de(
        individual, [8.0, 8.0], [4.0, 4.0], [0.0, 0.0], 1.0, 1.0, low=0.0, up=1.0
    )
    assert trial == [1.0, 1.0]


def test_mut_de_leaves_unwritten_genes_unclamped():
    tools.rng.seed(0)
    index = tools.rng.randrange(2)
    tools.rng.seed(0)
    individual: Any = [10.0, -5.0]
    (trial,) = tools.mut_de(
        individual, [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 1.0, 0.0, low=0.0, up=1.0
    )
    other = 1 - index
    assert 0.0 <= trial[index] <= 1.0
    assert trial[other] == [10.0, -5.0][other]
