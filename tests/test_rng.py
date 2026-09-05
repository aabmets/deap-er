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
import pytest
from deap_er import Checkpoint, tools


def test_seed_discards_the_uniform_buffer():
    tools.rng.seed(0)
    first = [tools.rng.random() for _ in range(5)]
    tools.rng.seed(0)
    for _ in range(3):
        tools.rng.random()
    tools.rng.seed(0)
    assert [tools.rng.random() for _ in range(5)] == first


def test_choice_sample_shuffle_keep_list_individuals():
    individuals = [[1, 2], [3, 4], [5, 6], [7, 8]]

    tools.rng.seed(1)
    picked = tools.rng.choice(individuals)
    assert picked in individuals
    assert picked in ([1, 2], [3, 4], [5, 6], [7, 8])

    tools.rng.seed(1)
    sampled = tools.rng.sample(individuals, k=2)
    assert len(sampled) == 2
    assert all(ind in individuals for ind in sampled)

    tools.rng.seed(1)
    shuffled = [[1, 2], [3, 4], [5, 6], [7, 8]]
    tools.rng.shuffle(shuffled)
    assert sorted(shuffled) == individuals
    assert all(ind in individuals for ind in shuffled)


def test_randint_is_inclusive_and_randrange_is_exclusive():
    tools.rng.seed(2)
    assert {tools.rng.randint(1, 1) for _ in range(8)} == {1}

    with pytest.raises(ValueError, match="empty range"):
        tools.rng.randrange(1, 1)

    tools.rng.seed(2)
    assert {tools.rng.randint(0, 2) for _ in range(200)} == {0, 1, 2}

    tools.rng.seed(2)
    assert {tools.rng.randrange(0, 2) for _ in range(200)} == {0, 1}


def test_checkpoint_restores_the_next_random(tmp_path):
    tools.rng.seed(42)
    for _ in range(10):
        tools.rng.random()

    cpt = Checkpoint(file_name="rng.dcpf", dir_path=tmp_path, autoload=False)
    cpt.save()
    expected = tools.rng.random()

    tools.rng.seed(99)
    cpt.load()
    assert tools.rng.random() == expected
