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
import contextvars
import time
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = [
    "bind_perf_nodeid",
    "measure_section",
    "take_perf_sections",
    "unbind_perf_nodeid",
]

_current_nodeid: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "perf_nodeid",
    default=None,
)
_sections: dict[str, dict[str, float]] = {}


def bind_perf_nodeid(nodeid: str | None) -> contextvars.Token[str | None]:
    return _current_nodeid.set(nodeid)


def unbind_perf_nodeid(token: contextvars.Token[str | None]) -> None:
    _current_nodeid.reset(token)


def take_perf_sections(nodeid: str) -> dict[str, float]:
    return _sections.pop(nodeid, {})


@contextmanager
def measure_section(label: str) -> Iterator[None]:
    if not label or not label.strip():
        raise ValueError("measure_section label must be a non-empty string")
    nodeid = _current_nodeid.get()
    if nodeid is None:
        raise RuntimeError("measure_section() must be used inside a running pytest test")
    bucket = _sections.setdefault(nodeid, {})
    if label in bucket:
        raise ValueError(f"measure_section label {label!r} is already used in {nodeid}")
    bucket[label] = 0.0
    started = time.perf_counter()
    try:
        yield
    finally:
        bucket[label] = time.perf_counter() - started
