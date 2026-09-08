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
NUMBA_FILES = frozenset({"test_window_pair.py", "test_window_ts.py", "test_promote_columnar.py"})
_warmed = False


def pytest_configure(config):
    cache = REPO / ".cache" / "numba"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache))


def _runs_numba(item: pytest.Item) -> bool:
    path = Path(getattr(item, "path", item.fspath))
    return "test_numba" in path.parts or path.name in NUMBA_FILES


def _warmup() -> None:
    if not numba_available():
        return
    compile_numba(parallel=True)
    compile_numba(parallel=True, dispatch=consumer_dispatch())
    gp.ema([0.0, 1.0, 2.0], 2)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    group = pytest.mark.xdist_group(name=NUMBA_GROUP)
    for item in items:
        if _runs_numba(item):
            item.add_marker(group)


def pytest_runtest_setup(item: pytest.Item) -> None:
    global _warmed
    if _warmed or not _runs_numba(item):
        return
    _warmed = True
    _warmup()


@pytest.fixture(scope="session")
def warmup_numba():
    _warmup()
