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

from .records.hall_of_fame import HallOfFame
from .various.rng import RNG

__all__: list[str] = ["CheckpointError", "read_checkpoint"]


class CheckpointError(ValueError):
    """A checkpoint file exists and is readable but cannot be loaded.

    Raised for a corrupt or truncated file, a pickle that references a
    type this process has not created, a payload that is not checkpoint
    state, a malformed RNG state, or a JSON ``hof`` without an
    individual type to rebuild it. Subclasses ``ValueError``.
    """


def read_checkpoint(file_path: Path, hof_ind_cls: type[Any] | None) -> dict[str, Any]:
    """Read and validate checkpoint state without changing global state.

    A ``hof`` stored as JSON is rebuilt with ``hof_ind_cls``, or with the
    individual type saved in the file when ``hof_ind_cls`` is None.

    Args:
        file_path: Checkpoint file to read.
        hof_ind_cls: Creator individual type used to rebuild ``hof``.

    Returns:
        The checkpoint attributes, ready to assign to the instance.

    Raises:
        OSError: If the file cannot be read.
        CheckpointError: If the file is not a loadable checkpoint.
    """
    with open(file_path, "rb") as f:
        try:
            # nosemgrep: python.lang.security.deserialization.pickle.avoid-dill
            state = dill.load(f)
        except OSError:
            raise
        except Exception as exc:
            raise _load_error(file_path, exc) from exc

    if not isinstance(state, dict):
        raise CheckpointError(
            f"{file_path} is not a loadable checkpoint: expected a dict, got {type(state).__name__}"
        )
    rng_state = state.get("_rng_state_")
    if rng_state is not None:
        try:
            RNG().set_state(rng_state)
        except Exception as exc:
            raise _load_error(file_path, exc) from exc
    if "_hof_json_" in state:
        _restore_hof(file_path, state, hof_ind_cls)
    return state


def _restore_hof(file_path: Path, state: dict[str, Any], hof_ind_cls: type[Any] | None) -> None:
    """Replace ``_hof_json_`` in ``state`` with the rebuilt ``hof``."""
    ind_cls = hof_ind_cls if hof_ind_cls is not None else state.get("_hof_ind_cls_")
    if ind_cls is None:
        raise CheckpointError(f"{file_path} stores hof as JSON; pass hof_ind_cls to load it")
    try:
        state["hof"] = HallOfFame.from_json(state.pop("_hof_json_"), ind_cls)
    except Exception as exc:
        raise _load_error(file_path, exc) from exc


def _load_error(file_path: Path, exc: Exception) -> CheckpointError:
    """Wrap a deserialisation failure in a ``CheckpointError``."""
    return CheckpointError(f"{file_path} is not a loadable checkpoint: {type(exc).__name__}: {exc}")
