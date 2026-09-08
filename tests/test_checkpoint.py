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
from pathlib import Path
from typing import Any, override

import dill
import pytest
from deap_er import Checkpoint, Fitness, creator, tools

HOF_CPT_FIT = "HOF_CPT_FIT"
HOF_CPT_IND = "HOF_CPT_IND"


@pytest.fixture
def hof_ind_cls():
    creator.create_type(HOF_CPT_FIT, Fitness, weights=(1.0,))
    creator.create_type(HOF_CPT_IND, list, fitness=creator.__dict__[HOF_CPT_FIT])
    yield creator.__dict__[HOF_CPT_IND]
    del creator.__dict__[HOF_CPT_FIT]
    del creator.__dict__[HOF_CPT_IND]


class TestCheckpoint:
    work_dir = Path(os.getcwd()).resolve().joinpath("qwerty")

    def test_file_name(self):
        _dir = self.work_dir
        cpt = Checkpoint(dir_path=_dir, autoload=False)
        assert cpt.file_path.suffix == ".dcpf"
        assert cpt.file_path.parent == _dir

    def test_dir_path(self):
        _dir = self.work_dir
        cpt = Checkpoint(file_name="asdfg.cpt", autoload=False)
        assert cpt.file_path.name == "asdfg.cpt"
        assert cpt.file_path.parent == _dir.with_name("deap-er")

    def test_saving(self, tmp_path):
        cpt1 = Checkpoint(file_name="asdfg.cpt", dir_path=tmp_path, autoload=False)
        cpt1.my_dict = {"key": "value"}

        assert not cpt1.file_path.exists()
        cpt1.save()
        assert cpt1.file_path.exists()

        cpt2 = Checkpoint(file_name="asdfg.cpt", dir_path=tmp_path, autoload=False)

        assert not hasattr(cpt2, "my_dict")
        cpt2.load()
        assert cpt2.my_dict == {"key": "value"}

    def test_range_1(self, tmp_path):
        cpt1 = Checkpoint(file_name="asdfg.cpt", dir_path=tmp_path, autoload=False)
        assert cpt1.last_op == "none"
        for i in cpt1.range(5):
            assert 0 < i < 6
        assert cpt1.last_op == "save_success"

        cpt2 = Checkpoint(file_name="asdfg.cpt", dir_path=tmp_path, autoload=True)
        assert cpt2.last_op == "load_success"
        for i in cpt2.range(5):
            assert 5 < i < 11
        assert cpt2.last_op == "save_success"

    def test_range_2(self, tmp_path):
        cpt = Checkpoint(file_name="asdfg.cpt", dir_path=tmp_path, autoload=False)
        cpt.save_freq = 0
        assert cpt.last_op == "none"
        for i in cpt.range(10):
            if i == 5:
                assert cpt.last_op == "save_success"

    def test_load_and_save_errors_without_raising(self, tmp_path):
        missing = Checkpoint(
            file_name="missing.dcpf", dir_path=tmp_path, autoload=False, raise_errors=False
        )
        assert missing.load() is False
        assert missing.last_op == "load_error"
        assert missing.is_loaded() is False

        blocked = tmp_path / "blocked.dcpf"
        blocked.mkdir()
        writer = Checkpoint(
            file_name="blocked.dcpf", dir_path=tmp_path, autoload=False, raise_errors=False
        )
        assert writer.save() is False
        assert writer.last_op == "save_error"
        assert writer.is_saved() is False

    def test_failed_save_leaves_previous_checkpoint_loadable(self, tmp_path):
        cpt = Checkpoint(file_name="keep.dcpf", dir_path=tmp_path, autoload=False)
        cpt.generation = 7
        assert cpt.save() is True

        class Boom:
            @override
            def __getstate__(self):
                raise TypeError("nope")

        cpt.payload = Boom()
        assert cpt.save() is False
        assert cpt.last_op == "save_error"

        loaded = Checkpoint(
            file_name="keep.dcpf", dir_path=tmp_path, autoload=False, raise_errors=False
        )
        assert loaded.load() is True
        assert loaded.generation == 7

        truncated = tmp_path / "trunc.dcpf"
        truncated.write_bytes(b"\x80\x04")
        broken = Checkpoint(
            file_name="trunc.dcpf", dir_path=tmp_path, autoload=False, raise_errors=False
        )
        assert broken.load() is False
        assert broken.last_op == "load_error"

    def test_load_keeps_constructor_path_and_raise_errors(self, tmp_path):
        dir_a = tmp_path / "A"
        dir_b = tmp_path / "B"
        dir_a.mkdir()
        dir_b.mkdir()
        source = Checkpoint(
            file_name="run.dcpf", dir_path=dir_a, autoload=False, raise_errors=False
        )
        source.generation = 7
        source.save()
        (dir_b / "run.dcpf").write_bytes((dir_a / "run.dcpf").read_bytes())

        loaded = Checkpoint(
            file_name="run.dcpf", dir_path=dir_b, autoload=False, raise_errors=True, make_dir=False
        )
        assert loaded.load() is True
        assert loaded.file_path == dir_b / "run.dcpf"
        assert loaded.raise_errors is True
        assert loaded.make_dir is False
        assert loaded.generation == 7

    def test_save_freq_can_be_disabled_during_range(self, tmp_path):
        cpt = Checkpoint(file_name="freq.dcpf", dir_path=tmp_path, autoload=False)
        saves = []
        original = cpt.save

        def counted() -> bool:
            saves.append(cpt._range_counter_)
            return original()

        hook: Any = cpt
        hook.save = counted
        cpt.save_freq = 0
        for gen in cpt.range(4):
            if gen == 2:
                cpt.save_freq = -1
        assert saves == [1]

        other = Checkpoint(file_name="freq2.dcpf", dir_path=tmp_path, autoload=False)
        other_saves = []
        other_original = other.save

        def other_counted() -> bool:
            other_saves.append(other._range_counter_)
            return other_original()

        other_hook: Any = other
        other_hook.save = other_counted
        other.save_freq = -1
        for gen in other.range(3):
            if gen == 1:
                other.save_freq = 0
        assert other_saves[-1] == 3
        assert len(other_saves) >= 1

    def test_range_disabled_and_rejects_negative(self, tmp_path):
        cpt = Checkpoint(file_name="nosave.dcpf", dir_path=tmp_path, autoload=False)
        cpt.save_freq = -1
        assert list(cpt.range(3)) == [1, 2, 3]
        assert cpt.last_op == "none"
        with pytest.raises(ValueError, match="negative"):
            list(cpt.range(-1))

    def test_hof_round_trips_as_json_not_dill(self, tmp_path, hof_ind_cls):
        hof = tools.HallOfFame(maxsize=2)
        for value in (3.0, 7.0, 1.0):
            ind = hof_ind_cls([int(value)])
            ind.fitness.values = (value,)
            hof.update([ind])

        writer = Checkpoint(
            file_name="hof.dcpf",
            dir_path=tmp_path,
            autoload=False,
            hof_ind_cls=hof_ind_cls,
        )
        writer.hof = hof
        assert writer.save() is True

        loaded = Checkpoint(
            file_name="hof.dcpf",
            dir_path=tmp_path,
            autoload=False,
            hof_ind_cls=hof_ind_cls,
        )
        assert loaded.load() is True
        assert [list(ind) for ind in loaded.hof] == [[7], [3]]
        assert [ind.fitness.values for ind in loaded.hof] == [(7.0,), (3.0,)]

        with open(tmp_path / "hof.dcpf", "rb") as f:
            payload = dill.load(f)
        assert "hof" not in payload
        assert '"maxsize": 2' in payload["_hof_json_"]

    def test_hof_json_saved_without_ind_cls_restored_only_when_provided(
        self, tmp_path, hof_ind_cls
    ):
        hof = tools.HallOfFame(maxsize=1)
        ind = hof_ind_cls([5])
        ind.fitness.values = (5.0,)
        hof.update([ind])

        writer = Checkpoint(file_name="noclazz.dcpf", dir_path=tmp_path, autoload=False)
        writer.hof = hof
        assert writer.save() is True

        with open(tmp_path / "noclazz.dcpf", "rb") as f:
            payload = dill.load(f)
        assert "hof" not in payload
        assert isinstance(payload["_hof_json_"], str)

        loaded_no_cls = Checkpoint(
            file_name="noclazz.dcpf", dir_path=tmp_path, autoload=False
        )
        assert loaded_no_cls.load() is True
        assert not hasattr(loaded_no_cls, "hof")
        assert "_hof_json_" in loaded_no_cls.__dict__

        loaded_with_cls = Checkpoint(
            file_name="noclazz.dcpf",
            dir_path=tmp_path,
            autoload=False,
            hof_ind_cls=hof_ind_cls,
        )
        assert loaded_with_cls.load() is True
        assert hasattr(loaded_with_cls, "hof")
        assert not hasattr(loaded_with_cls, "_hof_json_")
        assert list(loaded_with_cls.hof[0]) == [5]
        assert loaded_with_cls.hof[0].fitness.values == (5.0,)
