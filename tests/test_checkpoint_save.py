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
import os
import sys
from pathlib import Path
from typing import Any, override

import dill
import pytest
from deap_er import Checkpoint


class SaveDuringPickle:
    """Saves another checkpoint to the same file while being pickled."""

    def __init__(self, other: Checkpoint) -> None:
        self.other = other
        self.saved: bool | None = None

    @override
    def __reduce__(self) -> tuple[Any, ...]:
        self.saved = self.other.save()
        return int, (0,)


class ReduceRaises:
    @override
    def __reduce__(self) -> tuple[Any, ...]:
        raise ValueError("cannot pickle")


def _checkpoint(tmp_path: Path, **kwargs: Any) -> Checkpoint:
    return Checkpoint("c.dcpf", tmp_path, autoload=False, **kwargs)


def _files(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in tmp_path.iterdir())


def test_concurrent_writers_do_not_share_a_staging_file(tmp_path):
    inner = _checkpoint(tmp_path)
    inner.payload = "inner"
    outer = _checkpoint(tmp_path)
    outer.payload = "outer"
    outer.hook = SaveDuringPickle(inner)
    assert outer.save() is True
    assert outer.hook.saved is True
    assert _files(tmp_path) == ["c.dcpf"]
    loaded = _checkpoint(tmp_path, raise_errors=True)
    assert loaded.load() is True
    assert loaded.payload == "outer"


def test_save_fsyncs_file_and_directory(tmp_path, monkeypatch):
    synced: list[int] = []
    real_fsync = os.fsync

    def record(fd: int) -> None:
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", record)
    cp = _checkpoint(tmp_path)
    cp.payload = 1
    assert cp.save() is True
    assert len(synced) >= 2


def test_failed_replace_removes_staging_file(tmp_path, monkeypatch):
    def fail(*_args: Any) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail)
    cp = _checkpoint(tmp_path, raise_errors=True)
    with pytest.raises(OSError, match="replace failed"):
        cp.save()
    assert _files(tmp_path) == []


@pytest.mark.parametrize("raise_errors", [False, True])
def test_reduce_value_error_is_a_save_error(tmp_path, raise_errors):
    cp = _checkpoint(tmp_path, raise_errors=raise_errors)
    cp.bad = ReduceRaises()
    if raise_errors:
        with pytest.raises(ValueError, match="cannot pickle"):
            cp.save()
    else:
        assert cp.save() is False
        assert cp.last_op == "save_error"
    assert _files(tmp_path) == []


def test_deep_nesting_is_a_save_error(tmp_path):
    deep: list[Any] = []
    for _ in range(100_000):
        deep = [deep]
    cp = _checkpoint(tmp_path)
    cp.deep = deep
    assert cp.save() is False
    assert cp.last_op == "save_error"
    assert _files(tmp_path) == []


def test_plain_state_is_saved_without_dill(tmp_path, monkeypatch):
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("dill pickler used")

    monkeypatch.setattr(dill, "dump", refuse)
    monkeypatch.setattr(dill, "dumps", refuse)
    cp = _checkpoint(tmp_path, raise_errors=True)
    cp.payload = {"generation": 3, "values": [1.5, 2.5]}
    assert cp.save() is True
    loaded = _checkpoint(tmp_path, raise_errors=True)
    assert loaded.load() is True
    assert loaded.payload == {"generation": 3, "values": [1.5, 2.5]}


def test_lambda_falls_back_to_dill(tmp_path):
    cp = _checkpoint(tmp_path, raise_errors=True)
    cp.func = lambda x: x * 2
    assert cp.save() is True
    loaded = _checkpoint(tmp_path, raise_errors=True)
    assert loaded.load() is True
    assert loaded.func(4) == 8


def test_main_reference_falls_back_to_dill(tmp_path, monkeypatch):
    class MainThing:
        pass

    MainThing.__module__ = "__main__"
    MainThing.__qualname__ = "CheckpointMainThing"
    monkeypatch.setattr(sys.modules["__main__"], "CheckpointMainThing", MainThing, raising=False)
    calls: list[Any] = []
    real_dumps = dill.dumps

    def spy(obj: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(obj)
        return real_dumps(obj, *args, **kwargs)

    monkeypatch.setattr(dill, "dumps", spy)
    cp = _checkpoint(tmp_path, raise_errors=True)
    cp.thing = MainThing()
    assert cp.save() is True
    assert len(calls) == 1
