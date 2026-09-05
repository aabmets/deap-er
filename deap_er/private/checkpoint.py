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
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, override

import dill

from .various.rng import rng

__all__: list[str] = ["Checkpoint"]


class Checkpoint:
    """Save and load evolution progress with dill.

    Only attributes set on the checkpoint instance are written. The
    library RNG state (buffered uniforms and Generator) is persisted
    as well.
    The target file is chosen at construction.

    Args:
        file_name: Checkpoint file name. Defaults to a random UUID
            with a ``.dcpf`` extension.
        dir_path: Directory for the file. Defaults to
            ``<cwd>/deap-er``.
        autoload: If True and the file exists, load it during
            initialization. Defaults to True.
        make_dir: If True, create missing parent directories on
            save. Defaults to True.
        raise_errors: If True, propagate I/O and pickle errors.
            Otherwise ``load`` and ``save`` return False. Defaults
            to False.
    """

    _dir_ = "deap-er"  # Checkpoint Directory
    _ext_ = ".dcpf"  # [D]eaper [C]heck [P]oint [F]ile
    _omit_ = ["_last_op_"]

    _rng_state_: Any = None

    def __getattr__(self, name: str) -> Any:
        """Allow dynamically assigned checkpoint attributes."""
        raise AttributeError(name)

    @override
    def __setattr__(self, name: str, value: Any) -> None:
        """Set a checkpoint attribute, including user-defined names."""
        object.__setattr__(self, name, value)

    _range_counter_: int = 0
    _save_freq_: float = 60.0
    _last_op_: str = "none"

    def __init__(
        self,
        file_name: str | None = None,
        dir_path: Path | None = None,
        autoload: bool = True,
        make_dir: bool = True,
        raise_errors: bool = False,
    ) -> None:
        """See the class docstring for argument meanings."""
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

    def load(self) -> bool:
        """Load attributes from the checkpoint file onto this instance.

        Returns:
            True on success, False on failure when ``raise_errors`` is False.

        Raises:
            OSError: If the file cannot be read and ``raise_errors`` is True.
            dill.PickleError: If deserialization fails and ``raise_errors`` is True.
        """
        file_path = self.file_path
        raise_errors = self.raise_errors
        make_dir = self.make_dir
        try:
            with open(self.file_path, "rb") as f:
                # nosemgrep: python.lang.security.deserialization.pickle.avoid-dill
                self.__dict__ = dill.load(f)
            self.file_path = file_path
            self.raise_errors = raise_errors
            self.make_dir = make_dir
            if self._rng_state_ is not None:
                rng.set_state(self._rng_state_)
        except (OSError, dill.PickleError, EOFError, TypeError) as ex:
            self.file_path = file_path
            self.raise_errors = raise_errors
            self.make_dir = make_dir
            if self.raise_errors:
                raise ex
            self._last_op_ = "load_error"
            return False
        self._last_op_ = "load_success"
        return True

    def save(self) -> bool:
        """Write this instance's attributes to the checkpoint file.

        Overwrites an existing file. Creates parent directories when
        ``make_dir`` is True.

        Returns:
            True on success, False on failure when ``raise_errors`` is False.

        Raises:
            OSError: If the file cannot be written and ``raise_errors`` is True.
            dill.PickleError: If serialization fails and ``raise_errors`` is True.
        """
        try:
            self._rng_state_ = rng.get_state()
            if self.make_dir:
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
            _dict_ = vars(self).copy()
            for key in self._omit_:
                _dict_.pop(key, None)
            tmp_path = self.file_path.with_name(self.file_path.name + ".tmp")
            with open(tmp_path, "wb") as f:
                # nosemgrep: python.lang.security.deserialization.pickle.avoid-dill
                dill.dump(_dict_, f)
            os.replace(tmp_path, self.file_path)
        except (OSError, dill.PickleError, EOFError, TypeError) as ex:
            tmp_path = self.file_path.with_name(self.file_path.name + ".tmp")
            tmp_path.unlink(missing_ok=True)
            if self.raise_errors:
                raise ex
            self._last_op_ = "save_error"
            return False
        self._last_op_ = "save_success"
        return True

    def range(self, generations: int) -> Iterator[int]:
        """Yield generation indices, saving periodically when enabled.

        Continues from the last stored counter. When ``save_freq`` is
        not ``-1``, the checkpoint is written about every
        ``save_freq`` seconds and once more after the last yield.
        ``save_freq`` may be changed while iterating.

        Args:
            generations: Number of generations to yield.

        Returns:
            A generator of integer generation indices.

        Raises:
            ValueError: If ``generations`` is negative.
        """
        if generations < 0:
            raise ValueError("Iterations argument cannot be a negative number.")
        from_ = self._range_counter_ + 1
        to_excl = self._range_counter_ + generations + 1
        last_save = time.time()
        for i in range(from_, to_excl):
            yield i
            self._range_counter_ = i
            if self.save_freq == -1:
                continue
            now = time.time()
            if now - last_save >= self._save_freq_:
                last_save = now
                self.save()
        if self.save_freq != -1:
            self.save()

    @property
    def save_freq(self) -> float:
        """Seconds between automatic saves during ``range()``.

        ``-1`` disables saving. The default is 60 seconds.

        Returns:
            The current period in seconds.
        """
        return self._save_freq_

    @save_freq.setter
    def save_freq(self, value: int | float) -> None:
        """Set the automatic-save period in seconds.

        Args:
            value: Period in seconds, or ``-1`` to disable saving.
        """
        self._save_freq_ = float(value)

    @property
    def last_op(self) -> str:
        """Status of the last load or save.

        One of ``none``, ``load_success``, ``load_error``,
        ``save_success``, or ``save_error``.
        """
        return self._last_op_

    def is_loaded(self) -> bool:
        """Return whether the last operation was a successful load.

        Returns:
            True if ``last_op`` is ``load_success``.
        """
        return self._last_op_ == "load_success"

    def is_saved(self) -> bool:
        """Return whether the last operation was a successful save.

        Returns:
            True if ``last_op`` is ``save_success``.
        """
        return self._last_op_ == "save_success"
