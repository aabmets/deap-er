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
    automatically persist the states of the builtin :mod:`random` and :mod:`numpy.random` *RNG-s*.

    :param name: The name of the checkpoint file. If not given, a random
        UUID + :code:`.dcpf` extension is used.
    :param path: The path to the checkpoints directory.
        If not given, the current working directory + :code:`/deap-er` is used.
    :param autoload: If True **and** the checkpoint file exists,
        load the file on initialization, optional.
        The default value is True.
    """
    # -------------------------------------------------------- #
    _dir = 'deap-er'  # Checkpoint Directory
    _ext = '.dcpf'    # [D]eaper [C]heck [P]oint [F]ile

    _rand_state: object = None
    _np_state: dict = None
    _counter: int = 0

    # -------------------------------------------------------- #
    def __init__(self,
                 name: Optional[str] = None,
                 path: Optional[Path] = None,
                 autoload: Optional[bool] = True):

        if name is None:
            name = str(uuid.uuid4()) + self._ext
        if path is None:
            path = Path(os.getcwd()).resolve()
            path = path.joinpath(self._dir)

        self.file_path = path.joinpath(name)
        if autoload is True:
            self.load()

    # -------------------------------------------------------- #
    def load(self, raise_errors: Optional[bool] = False) -> bool:
        """
        Loads the contents of the checkpoint file into :code:`self.__dict__`.

        :param raise_errors: If True, errors are propagated, optional.
            By default, errors are not propagated and False is returned instead.
        :raise IOError: If the operation failed and **raise_errors** is True.
        :raise dill.PickleError: If the operation failed and **raise_errors** is True.
        :return: True if the operation completed successfully.
        """
        if self.file_path.exists():
            try:
                with open(self.file_path, 'rb') as f:
                    self.__dict__ = dill.load(file=f)
            except (IOError, dill.PickleError) as ex:
                if raise_errors:
                    raise ex
            else:
                random.setstate(self._rand_state)
                np.random.set_state(self._np_state)
                return True
        return False

    # -------------------------------------------------------- #
    def save(self, raise_errors: Optional[bool] = False) -> bool:
        """
        Saves the contents of :code:`self.__dict__` into the checkpoint file.
        If the file already exists, it will be overwritten.
        If the target directory does not exist, it will be created recursively.

        :param raise_errors: If True, errors are propagated, optional.
            By default, errors are not propagated and False is returned instead.
        :raise IOError: If the operation failed and **raise_errors** is True.
        :raise dill.PickleError: If the operation failed and **raise_errors** is True.
        :return: True if the operation completed successfully.
        """
        self._rand_state = random.getstate()
        self._np_state = np.random.get_state()
        try:
            if not self.file_path.parent.exists():
                self.file_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )
            with open(self.file_path, 'wb') as f:
                dill.dump(
                    obj=self.__dict__, file=f,
                    protocol=dill.HIGHEST_PROTOCOL,
                    recurse=True
                )
        except (IOError, dill.PickleError) as ex:
            if raise_errors:
                raise ex
            return False
        return True

    # -------------------------------------------------------- #
    def range(self, stop: int, save_freq: int) -> int:
        """
        A special generator method that behaves almost like the builtin
        :code:`range()` function, but the attributes of the checkpoint object
        are automatically saved into the checkpoint file every **save_freq**
        iterations. The start value is automatically determined depending on
        the current state of the checkpoint object.

        :param stop: The stop value of the iterator, exclusive.
        :param save_freq: The frequency at which the checkpoint is saved to file.
        :return: A generator that yields integer values up to **stop**, exclusive.
        """
        if stop < 0:
            raise ValueError('Iterator stop value cannot be a negative number.')
        elif stop < self._counter:
            raise ValueError('Iterator current counter must be <= stop value.')

        itr = iter(range(self._counter, stop + 1))
        next(itr)
        try:
            while True:
                self._counter = next(itr)
                if self._counter % save_freq == 0:
                    self.save()
                yield self._counter
        except StopIteration:
            pass
