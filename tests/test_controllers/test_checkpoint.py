#
#   MIT License
#   
#   Copyright (c) 2022, Mattias Aabmets
#   
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#   
#   SPDX-License-Identifier: MIT
#
from deap_er import env
from pathlib import Path
import time
import os


# ====================================================================================== #
class TestCheckpoint:

    work_dir = Path(os.getcwd()).resolve().joinpath('qwerty')

    def test_file_name(self):
        _dir = self.work_dir
        cpt = env.Checkpoint(
            dir_path=_dir,
            autoload=False
        )
        assert cpt.file_path.suffix == '.dcpf'
        assert cpt.file_path.parent == _dir

    # -------------------------------------------------------- #
    def test_dir_path(self):
        _dir = self.work_dir
        cpt = env.Checkpoint(
            file_name='asdfg.cpt',
            autoload=False
        )
        assert cpt.file_path.name == 'asdfg.cpt'
        assert cpt.file_path.parent == _dir.with_name('deap-er')

    # -------------------------------------------------------- #
    def test_saving(self, tmp_path):
        cpt1 = env.Checkpoint(
            file_name='asdfg.cpt',
            dir_path=tmp_path,
            autoload=False
        )
        cpt1.my_dict = {'key': 'value'}

        assert not cpt1.file_path.exists()
        cpt1.save()
        assert cpt1.file_path.exists()

        cpt2 = env.Checkpoint(
            file_name='asdfg.cpt',
            dir_path=tmp_path,
            autoload=False
        )

        assert not hasattr(cpt2, 'my_dict')
        cpt2.load()
        assert getattr(cpt2, 'my_dict') == {'key': 'value'}

    # -------------------------------------------------------- #
    def test_range_1(self, tmp_path):
        cpt1 = env.Checkpoint(
            file_name='asdfg.cpt',
            dir_path=tmp_path,
            autoload=False
        )
        assert cpt1.last_op == 'none'
        for i in cpt1.range(5):
            assert 0 < i < 6
        assert cpt1.last_op == 'save_success'

        cpt2 = env.Checkpoint(
            file_name='asdfg.cpt',
            dir_path=tmp_path,
            autoload=True
        )
        assert cpt2.last_op == 'load_success'
        for i in cpt2.range(5):
            assert 5 < i < 11
        assert cpt2.last_op == 'save_success'

    # -------------------------------------------------------- #
    def test_range_2(self, tmp_path):
        cpt = env.Checkpoint(
            file_name='asdfg.cpt',
            dir_path=tmp_path,
            autoload=False
        )
        cpt.save_freq = 0.1
        assert cpt.last_op == 'none'
        for i in cpt.range(10):
            time.sleep(0.021)
            if i == 5:
                assert cpt.last_op == 'save_success'
