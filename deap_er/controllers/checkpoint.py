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
from typing import Optional
from pathlib import Path
import numpy as np
import random
import uuid
import dill
import os


__all__ = ['Checkpoint']


# ====================================================================================== #
class Checkpoint:
    """
    This class can be used to save and load evolution progress to and from files.
    It's implemented as a lightweight wrapper around the builtin :code:`open()` function.
    Objects are (de-)serialized using the `dill <https://pypi.org/project/dill/>`_ library.
    The target checkpoint file is assigned on instantiation. Checkpoint objects also
    automatically persist the *RNG* states of the :mod:`random` and :mod:`numpy.random` modules.

    :param file_name: The name of the checkpoint file.
        By default, a random UUID + :code:`.dcpf` extension is used.
    :param dir_path: The path to the checkpoints directory. By default,
        the current working directory + :code:`/deap-er` is used.
    :param autoload: If True **and** the checkpoint file exists, loads the
        file during initialization, optional. The default value is True.
    :param make_dir: If True, the target directory is recursively created
        on save, if it does not exist. The default value is True.
    :param raise_errors: If True, errors are propagated, optional.
        By default, errors are not propagated and False is returned instead.
    """
    # -------------------------------------------------------- #
    _dir_ = 'deap-er'  # Checkpoint Directory
    _ext_ = '.dcpf'    # [D]eaper [C]heck [P]oint [F]ile
    _omit_ = ['_last_op_']

    _rand_state_: object = None
    _numpy_state_: dict = None
    _range_counter_: int = 0
    _last_op_: str = 'none'

    # -------------------------------------------------------- #
    def __init__(self,
                 file_name: Optional[str] = None,
                 dir_path: Optional[Path] = None,
                 autoload: Optional[bool] = True,
                 make_dir: Optional[bool] = True,
                 raise_errors: Optional[bool] = False):
        if file_name is None:
            file_name = str(uuid.uuid4()) + self._ext_
        if dir_path is None:
            dir_path = Path(os.getcwd()).resolve()
            dir_path = dir_path.joinpath(self._dir_)
        self.file_path = dir_path.joinpath(file_name)
        self.raise_errors = raise_errors
        self.make_dir = make_dir
        if autoload is True:
            self.load()

    # -------------------------------------------------------- #
    def load(self) -> bool:
        """
        Loads the contents of the checkpoint file into :code:`self.__dict__`.

        :raise IOError: If the operation failed and :code:`self.raise_errors` is True.
        :raise dill.PickleError: If the operation failed and :code:`self.raise_errors` is True.
        :return: True if the operation completed successfully, False otherwise.
        """
        try:
            with open(self.file_path, 'rb') as f:
                self.__dict__ = dill.load(f)
            random.setstate(self._rand_state_)
            np.random.set_state(self._numpy_state_)
        except (IOError, dill.PickleError) as ex:
            if self.raise_errors:
                raise ex
            self._last_op_ = 'load_error'
            return False
        self._last_op_ = 'load_success'
        return True

    # -------------------------------------------------------- #
    def save(self) -> bool:
        """
        Saves the contents of :code:`self.__dict__` into the checkpoint file.
        If the file already exists, it will be overwritten.
        If the target directory does not exist, it will be created recursively.

        :raise IOError: If the operation failed and :code:`self.raise_errors` is True.
        :raise dill.PickleError: If the operation failed and :code:`self.raise_errors` is True.
        :return: True if the operation completed successfully, False otherwise.
        """
        try:
            self._rand_state_ = random.getstate()
            self._numpy_state_ = np.random.get_state()
            if self.make_dir:
                self.file_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )
            with open(self.file_path, 'wb') as f:
                _dict_ = vars(self).copy()
                for key in self._omit_:
                    _dict_.pop(key, None)
                dill.dump(_dict_, f)
        except (IOError, dill.PickleError) as ex:
            if self.raise_errors:
                raise ex
            self._last_op_ = 'save_error'
            return False
        self._last_op_ = 'save_success'
        return True

    # -------------------------------------------------------- #
    def range(self, iterations: int, save_freq: int) -> range:
        """
        A special generator method that behaves almost like the builtin
        :code:`range()` function, but the checkpoint object is automatically
        saved into the checkpoint file every **save_freq** iterations. The
        start and stop values are automatically determined from the current
        *(loaded)* state of the checkpoint object.

        :param iterations: The count of iterations to loop over.
            Each iteration, the internal counter is incremented by 1.
        :param save_freq: The frequency at which the checkpoint is saved to file.
        :return: A generator that yields the values of the internal counter.
        """
        if iterations < 0:
            raise ValueError(
                'Iterations argument cannot be a negative number.'
            )
        start = self._range_counter_ + 1
        stop = self._range_counter_ + iterations + 1
        for i in range(start, stop):
            self._range_counter_ = i
            if i % save_freq == 0:
                self.save()
            yield i

    # -------------------------------------------------------- #
    @property
    def last_op(self) -> str:
        """
        | Returns the status of the last operation performed on the checkpoint object.
        | Possible string-type return values are:

            * *none*
            * *load_success*
            * *load_error*
            * *save_success*
            * *save_error*
        """
        return self._last_op_
