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
import sys
from pathlib import Path

import pytest
from deap_er import gp
from deap_er.private.programming.numba.numba_ops import (
    numba_available,
)
from deap_er.private.programming.numba.numba_ops import (
    warmup_numba as compile_numba,
)
from tests.harness.numba_dispatch import consumer_dispatch

__all__ = [
    "pytest_collection_modifyitems",
    "pytest_configure",
    "pytest_runtest_setup",
    "warmup_numba",
]

REPO = Path(__file__).resolve().parents[2]
NUMBA_GROUP = "numba"
CONSUMER_GROUP = "numba_consumer"
NUMBA_FILES = frozenset({"test_window_pair.py", "test_window_ts.py", "test_promote_columnar.py"})
# Kernels that take a dispatcher cannot be cached on disk, so every
# process recompiles them. Consumer-dispatch tests run in their own
# xdist group so that compile overlaps the idle-dispatch one.
CONSUMER_FILES = frozenset({"test_numba_ops.py", "test_numba_batch_kernels.py"})
_warmed = set()


def pytest_configure(config):
    cache = REPO / ".cache" / "numba"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache))
    if numba_available():
        _untrace_numba_compiler()


def _untrace_numba_compiler() -> None:
    # Numba's compiler is pure Python, and line tracing by coverage adds
    # about 10 s to a cold compile. Only numba internals run untraced:
    # deap_er code around the compile is still measured.
    from numba.core.dispatcher import Dispatcher

    compile_traced = Dispatcher.compile

    def compile_untraced(self, sig):
        tracer = sys.gettrace()
        sys.settrace(None)
        try:
            return compile_traced(self, sig)
        finally:
            sys.settrace(tracer)

    Dispatcher.compile = compile_untraced


def _runs_numba(item: pytest.Item) -> bool:
    path = Path(getattr(item, "path", item.fspath))
    return "test_numba" in path.parts or path.name in NUMBA_FILES


def _group(item: pytest.Item) -> str:
    path = Path(getattr(item, "path", item.fspath))
    return CONSUMER_GROUP if path.name in CONSUMER_FILES else NUMBA_GROUP


def _warmup(group: str) -> None:
    if not numba_available():
        return
    if group == CONSUMER_GROUP:
        compile_numba(parallel=True, dispatch=consumer_dispatch())
        return
    compile_numba(parallel=True)
    gp.ema([0.0, 1.0, 2.0], 2)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        if _runs_numba(item):
            item.add_marker(pytest.mark.xdist_group(name=_group(item)))


def pytest_runtest_setup(item: pytest.Item) -> None:
    if not _runs_numba(item):
        return
    group = _group(item)
    if group in _warmed:
        return
    _warmed.add(group)
    _warmup(group)


@pytest.fixture(scope="session")
def warmup_numba():
    _warmup(NUMBA_GROUP)
    _warmup(CONSUMER_GROUP)
