# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 - Mattias Aabmets, The DEAP Team and Other Contributors           #
#                                                                                        #
#   Permission is hereby granted, free of charge, to any person obtaining a copy         #
#   of this software and associated documentation files (the "Software"), to deal        #
#   in the Software without restriction, including without limitation the rights         #
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell            #
#   copies of the Software, and to permit persons to whom the Software is                #
#   furnished to do so, subject to the following conditions:                             #
#                                                                                        #
#   The above copyright notice and this permission notice shall be included in all       #
#   copies or substantial portions of the Software.                                      #
#                                                                                        #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR           #
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,             #
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE          #
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER               #
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,        #
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE        #
#   SOFTWARE.                                                                            #
#                                                                                        #
# ====================================================================================== #
from deap_er import env
from pathlib import Path
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
    def test_range(self, tmp_path):
        cpt1 = env.Checkpoint(
            file_name='asdfg.cpt',
            dir_path=tmp_path,
            autoload=False
        )
        assert cpt1.last_op == 'none'
        for i in cpt1.range(5, 5):
            assert 0 < i < 6
        assert cpt1.last_op == 'save_success'

        cpt2 = env.Checkpoint(
            file_name='asdfg.cpt',
            dir_path=tmp_path,
            autoload=True
        )
        assert cpt2.last_op == 'load_success'
        for i in cpt2.range(5, 5):
            assert 5 < i < 11
        assert cpt2.last_op == 'save_success'
