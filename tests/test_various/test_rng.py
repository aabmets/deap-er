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
    with pytest.raises(ValueError, match="low >= high"):
        tools.rng.randint(5, 3)
    with pytest.raises(OverflowError, match="64-bit"):
        tools.rng.randint(0, 2**64)

    tools.rng.seed(2)
    assert {tools.rng.randint(0, 2) for _ in range(200)} == {0, 1, 2}

    tools.rng.seed(2)
    assert {tools.rng.randrange(0, 2) for _ in range(200)} == {0, 1}


def test_seed_discards_the_integer_buffer():
    tools.rng.seed(0)
    first = [tools.rng.randint(0, 10) for _ in range(5)]
    tools.rng.seed(0)
    for _ in range(3):
        tools.rng.randint(0, 10)
    tools.rng.seed(0)
    assert [tools.rng.randint(0, 10) for _ in range(5)] == first


def test_choice_rejects_empty_sequence():
    with pytest.raises(IndexError, match="empty"):
        tools.rng.choice([])


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


def test_checkpoint_restores_the_next_choice(tmp_path):
    population = list(range(20))
    tools.rng.seed(42)
    for _ in range(10):
        tools.rng.choice(population)

    cpt = Checkpoint(file_name="rng-choice.dcpf", dir_path=tmp_path, autoload=False)
    cpt.save()
    expected = tools.rng.choice(population)

    tools.rng.seed(99)
    cpt.load()
    assert tools.rng.choice(population) == expected


def test_checkpoint_restores_the_next_randint(tmp_path):
    tools.rng.seed(42)
    for _ in range(10):
        tools.rng.randint(0, 20)

    cpt = Checkpoint(file_name="rng-randint.dcpf", dir_path=tmp_path, autoload=False)
    cpt.save()
    expected = tools.rng.randint(0, 20)

    tools.rng.seed(99)
    cpt.load()
    assert tools.rng.randint(0, 20) == expected


def test_checkpoint_restores_mixed_leftovers(tmp_path):
    tools.rng.seed(42)
    for _ in range(5):
        tools.rng.random()
        tools.rng.randint(0, 20)

    cpt = Checkpoint(file_name="rng-mixed.dcpf", dir_path=tmp_path, autoload=False)
    cpt.save()
    expected_float = tools.rng.random()
    expected_int = tools.rng.randint(0, 20)

    tools.rng.seed(99)
    cpt.load()
    assert tools.rng.random() == expected_float
    assert tools.rng.randint(0, 20) == expected_int


def test_integers_shares_the_uint64_buffer_with_randint_and_choice():
    tools.rng.seed(3)
    via_integers = [tools.rng.integers(0, 10) for _ in range(5)]
    tools.rng.seed(3)
    via_randint = [tools.rng.randint(0, 9) for _ in range(5)]
    tools.rng.seed(3)
    via_choice = [tools.rng.choice(range(10)) for _ in range(5)]
    assert via_integers == via_randint == via_choice


def test_sized_integers_drop_integer_leftover_and_keep_floats():
    tools.rng.seed(0)
    tools.rng.random()
    tools.rng.randint(0, 9)
    state = tools.rng.get_state()
    leftover_int = tools.rng.randint(0, 9)
    tools.rng.set_state(state)
    leftover_float = tools.rng.random()

    tools.rng.set_state(state)
    tools.rng.integers(0, 10, size=5)
    assert tools.rng.randint(0, 9) != leftover_int
    tools.rng.set_state(state)
    tools.rng.integers(0, 10, size=5)
    assert tools.rng.random() == leftover_float


def test_unused_integer_buffer_is_zeroed():
    tools.rng.seed(0)
    tools.rng.randint(0, 3)
    tools.rng.seed(1)
    state = tools.rng.get_state()
    assert state["iindex"] == 1024
    assert not any(int(value) for value in state["ibuf"])


def test_set_state_legacy_keys_drop_integer_leftover():
    tools.rng.seed(7)
    for _ in range(8):
        tools.rng.randint(0, 9)
    full = tools.rng.get_state()
    leftover = tools.rng.randint(0, 9)

    tools.rng.set_state(full)
    assert tools.rng.randint(0, 9) == leftover

    legacy = {key: full[key] for key in ("bit_generator", "buf", "index")}
    tools.rng.set_state(legacy)
    assert tools.rng.randint(0, 9) != leftover
