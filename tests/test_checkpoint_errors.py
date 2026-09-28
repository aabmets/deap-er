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
from pathlib import Path
from typing import Any

import dill
import pytest
from deap_er import Checkpoint, CheckpointError, Fitness, creator, tools

GONE_IND = "CPT_ERR_GONE_IND"
HOF_FIT = "CPT_ERR_HOF_FIT"
HOF_IND = "CPT_ERR_HOF_IND"


@pytest.fixture
def hof_ind_cls():
    creator.create_type(HOF_FIT, Fitness, weights=(1.0,))
    creator.create_type(HOF_IND, list, fitness=creator.__dict__[HOF_FIT])
    yield creator.__dict__[HOF_IND]
    del creator.__dict__[HOF_FIT]
    del creator.__dict__[HOF_IND]


def _hof(ind_cls: type[Any]) -> tools.HallOfFame:
    hof = tools.HallOfFame(maxsize=2)
    for value in (3.0, 7.0):
        ind = ind_cls([int(value)])
        ind.fitness.values = (value,)
        hof.update([ind])
    return hof


def _dump(path: Path, payload: Any) -> None:
    with open(path, "wb") as f:
        dill.dump(payload, f)


def _write_missing_type(path: Path) -> None:
    creator.create_type(GONE_IND, list)
    try:
        _dump(path, {"member": creator.__dict__[GONE_IND]([1])})
    finally:
        del creator.__dict__[GONE_IND]


def _write_bad_rng(path: Path) -> None:
    _dump(path, {"generation": 9, "_rng_state_": {"bit_generator": "garbage"}})


BROKEN_FILES = {
    "empty": lambda path: path.write_bytes(b""),
    "truncated": lambda path: path.write_bytes(b"\x80\x04"),
    "not_a_dict": lambda path: _dump(path, [1, 2, 3]),
    "missing_creator_type": _write_missing_type,
    "bad_rng_state": _write_bad_rng,
    "legacy_json_hof_without_type": lambda path: _dump(path, {"_hof_json_": "{}"}),
}


def test_autoload_with_raise_errors_on_missing_file(tmp_path):
    cpt = Checkpoint(dir_path=tmp_path, autoload=True, raise_errors=True)
    assert cpt.last_op == "none"
    with pytest.raises(FileNotFoundError):
        cpt.load()


def test_checkpoint_error_is_a_value_error():
    assert issubclass(CheckpointError, ValueError)


@pytest.mark.parametrize("kind", sorted(BROKEN_FILES))
def test_broken_file_raises_checkpoint_error(tmp_path, kind):
    BROKEN_FILES[kind](tmp_path / "bad.dcpf")
    cpt = Checkpoint(file_name="bad.dcpf", dir_path=tmp_path, autoload=False, raise_errors=True)
    with pytest.raises(CheckpointError, match="bad.dcpf"):
        cpt.load()


@pytest.mark.parametrize("kind", sorted(BROKEN_FILES))
def test_broken_file_returns_false(tmp_path, kind):
    BROKEN_FILES[kind](tmp_path / "bad.dcpf")
    cpt = Checkpoint(file_name="bad.dcpf", dir_path=tmp_path, autoload=True)
    assert cpt.last_op == "load_error"
    assert cpt.load() is False


def test_unreadable_path_raises_os_error(tmp_path):
    (tmp_path / "dir.dcpf").mkdir()
    cpt = Checkpoint(file_name="dir.dcpf", dir_path=tmp_path, autoload=False, raise_errors=True)
    with pytest.raises(OSError):
        cpt.load()
    quiet = Checkpoint(file_name="dir.dcpf", dir_path=tmp_path, autoload=False)
    assert quiet.load() is False


@pytest.mark.parametrize("kind", ["bad_rng_state", "legacy_json_hof_without_type"])
def test_failed_load_changes_neither_instance_nor_rng(tmp_path, kind):
    BROKEN_FILES[kind](tmp_path / "bad.dcpf")
    cpt = Checkpoint(file_name="bad.dcpf", dir_path=tmp_path, autoload=False)
    cpt.generation = 3
    before = {key: value for key, value in vars(cpt).items() if key != "_last_op_"}
    tools.rng.seed(5)
    rng_state = tools.rng.get_state()

    assert cpt.load() is False

    after = {key: value for key, value in vars(cpt).items() if key != "_last_op_"}
    assert after == before
    drawn = tools.rng.random()
    tools.rng.set_state(rng_state)
    assert tools.rng.random() == drawn


def test_hof_without_ind_cls_round_trips_with_dill(tmp_path, hof_ind_cls):
    writer = Checkpoint(file_name="hof.dcpf", dir_path=tmp_path, autoload=False)
    writer.hof = _hof(hof_ind_cls)
    assert writer.save() is True

    with open(tmp_path / "hof.dcpf", "rb") as f:
        payload = dill.load(f)
    assert "_hof_json_" not in payload

    loaded = Checkpoint(file_name="hof.dcpf", dir_path=tmp_path, autoload=False)
    assert loaded.load() is True
    assert isinstance(loaded.hof, tools.HallOfFame)
    assert [list(ind) for ind in loaded.hof] == [[7], [3]]
    assert "_hof_json_" not in vars(loaded)


def test_legacy_json_hof_loads_when_ind_cls_given(tmp_path, hof_ind_cls):
    payload = {"generation": 4, "_hof_json_": _hof(hof_ind_cls).to_json()}
    _dump(tmp_path / "old.dcpf", payload)

    loaded = Checkpoint(
        file_name="old.dcpf", dir_path=tmp_path, autoload=False, hof_ind_cls=hof_ind_cls
    )
    assert loaded.load() is True
    assert [list(ind) for ind in loaded.hof] == [[7], [3]]
    assert loaded.generation == 4

    without = Checkpoint(file_name="old.dcpf", dir_path=tmp_path, autoload=False, raise_errors=True)
    with pytest.raises(CheckpointError, match="pass hof_ind_cls"):
        without.load()


def test_json_hof_loads_with_the_saved_ind_cls(tmp_path, hof_ind_cls):
    writer = Checkpoint(
        file_name="hof.dcpf", dir_path=tmp_path, autoload=False, hof_ind_cls=hof_ind_cls
    )
    writer.hof = _hof(hof_ind_cls)
    assert writer.save() is True

    loaded = Checkpoint(file_name="hof.dcpf", dir_path=tmp_path, autoload=False)
    assert loaded.load() is True
    assert [list(ind) for ind in loaded.hof] == [[7], [3]]
    assert isinstance(loaded.hof[0], hof_ind_cls)
