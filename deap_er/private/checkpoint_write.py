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
import pickle
import uuid
from pathlib import Path
from typing import Any, cast

import dill

__all__: list[str] = ["serialize_state", "write_atomic"]


def serialize_state(state: dict[str, Any]) -> bytes:
    """Pickle checkpoint state, preferring the fast C pickler.

    dill's pickler is pure Python and many times slower. The C pickler
    is tried first, and dill is used only when it fails (lambdas, local
    functions) or when the result references ``__main__``, which dill
    pickles by value instead. ``dill.load`` reads either output.

    Args:
        state: The checkpoint attributes to serialize.

    Returns:
        The pickled state.

    Raises:
        Exception: Whatever dill raises when it cannot pickle ``state``.
    """
    try:
        data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        data = None
    if data is not None and b"__main__" not in data:
        return data
    # nosemgrep: python.lang.security.deserialization.pickle.avoid-dill
    return cast(bytes, dill.dumps(state))


def write_atomic(file_path: Path, data: bytes) -> None:
    """Durably replace ``file_path`` with ``data``.

    The bytes go to a staging file with a unique name in the same
    directory, so concurrent writers never share one. The staging file
    and then the directory are fsynced around the rename, so the new
    file survives a power loss. A failed write removes the staging file.

    Args:
        file_path: The file to replace.
        data: The new file contents.

    Raises:
        OSError: If the file cannot be written.
    """
    tmp_path = file_path.with_name(f"{file_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp_path, "xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    _fsync_dir(file_path.parent)


def _fsync_dir(dir_path: Path) -> None:
    """Flush a directory entry change to disk where the OS allows it."""
    if not hasattr(os, "O_DIRECTORY"):
        return
    fd = os.open(dir_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
