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
import multiprocessing
from collections.abc import Iterable, Sequence
from typing import Any

import numpy
import pytest
from deap_er import Checkpoint, tools


def draw_pair(_item: object) -> tuple[float, int]:
    return tools.rng.random(), tools.rng.randint(0, 99)


def reversed_map(func: Any, items: Iterable[Any]) -> list[Any]:
    return [func(item) for item in reversed(list(items))]


def _stream_floats(seed: int | Sequence[int], worker_id: int, n: int = 8) -> list[float]:
    child = tools.spawn_rng(seed, worker_id)
    return [child.random() for _ in range(n)]


def _stream_ints(seed: int | Sequence[int], worker_id: int, n: int = 4) -> list[int]:
    child = tools.spawn_rng(seed, worker_id)
    return [child.randint(0, 20) for _ in range(n)]


def test_spawn_rng_is_deterministic_and_independent_of_parent():
    tools.rng.seed(11)
    parent = [tools.rng.random() for _ in range(8)]
    child_a = _stream_floats(11, 0)
    child_b = _stream_floats(11, 1)
    again = _stream_floats(11, 0)
    tools.rng.seed(11)
    after_spawn = [tools.rng.random() for _ in range(8)]
    assert child_a == again
    assert child_a != child_b
    assert child_a != parent
    assert child_b != parent
    assert after_spawn == parent


def test_spawn_rng_accepts_sequence_seed_and_numpy_worker_id():
    first = _stream_ints([3, 5], 2)
    second = _stream_ints((3, 5), 2)
    other = _stream_ints([3, 5], 3)
    assert first == second
    assert first != other
    numpy_id: Any = numpy.int64(0)
    assert _stream_floats(4, numpy_id) == _stream_floats(4, 0)


def test_spawn_rng_rejects_bad_seed_and_worker_id():
    bad_seed: Any = "11"
    bad_worker: Any = 0.5
    with pytest.raises(TypeError, match="sequence of ints"):
        tools.spawn_rng(bad_seed, 0)
    with pytest.raises(ValueError, match="empty"):
        tools.spawn_rng([], 0)
    with pytest.raises(TypeError, match="worker_id"):
        tools.spawn_rng(1, bad_worker)
    with pytest.raises(ValueError, match="non-negative"):
        tools.spawn_rng(1, -1)


def test_bind_spawned_rng_installs_the_child_on_the_process_wide_rng():
    child = tools.spawn_rng(21, 4)
    expected = [child.random() for _ in range(5)]
    tools.rng.seed(0)
    tools.bind_spawned_rng(21, 4)
    assert [tools.rng.random() for _ in range(5)] == expected


def test_map_spawned_matches_across_completion_order():
    items = list(range(8))
    sequential = tools.map_spawned(draw_pair, items, seed=7)
    reversed_done = tools.map_spawned(draw_pair, items, seed=7, map_func=reversed_map)
    again = tools.map_spawned(draw_pair, items, seed=7)
    assert sequential == reversed_done == again
    assert len(set(sequential)) == len(sequential)


def test_map_spawned_restores_the_parent_stream():
    tools.rng.seed(13)
    before = tools.rng.get_state()
    tools.map_spawned(draw_pair, range(4), seed=13)
    after = [tools.rng.random() for _ in range(4)]
    tools.rng.set_state(before)
    assert [tools.rng.random() for _ in range(4)] == after


def test_map_spawned_restores_the_parent_when_func_raises():
    tools.rng.seed(5)
    before = tools.rng.get_state()

    def boom(_item: object) -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        tools.map_spawned(boom, [1], seed=5)
    after = [tools.rng.random() for _ in range(3)]
    tools.rng.set_state(before)
    assert [tools.rng.random() for _ in range(3)] == after


def test_map_spawned_empty_and_incomplete_map():
    assert tools.map_spawned(draw_pair, [], seed=1) == []

    def drop_last(func: Any, items: Iterable[Any]) -> list[Any]:
        return [func(item) for item in list(items)[:-1]]

    with pytest.raises(ValueError, match="one \\(worker_id, value\\)"):
        tools.map_spawned(draw_pair, range(3), seed=1, map_func=drop_last)


def test_checkpoint_parent_is_unchanged_by_spawned_workers(tmp_path):
    tools.rng.seed(42)
    for _ in range(6):
        tools.rng.random()
    cpt = Checkpoint(file_name="spawn-parent.dcpf", dir_path=tmp_path, autoload=False)
    cpt.save()
    expected = tools.rng.random()
    tools.map_spawned(draw_pair, range(5), seed=42)
    tools.rng.seed(99)
    cpt.load()
    assert tools.rng.random() == expected


def test_process_pool_matches_in_process_map():
    items = list(range(6))
    expected = tools.map_spawned(draw_pair, items, seed=17)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(2) as pool:
        got = tools.map_spawned(draw_pair, items, seed=17, map_func=pool.map)
    assert got == expected
