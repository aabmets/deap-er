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
from deap_er.private.programming.numba.numba_compile import ensure_numba_cache_dir


def test_warmup_numba_is_exported_from_gp():
    assert callable(gp.warmup_numba)


@pytest.mark.xdist_group(name="numba")
def test_ensure_numba_cache_dir_uses_a_stable_absolute_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    other_cwd = tmp_path / "other_cwd"
    other_cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.chdir(other_cwd)

    ensure_numba_cache_dir()

    cache = Path(os.environ["NUMBA_CACHE_DIR"])
    expected = (home / ".cache" / "deap-er" / "numba").resolve()
    assert cache.is_absolute()
    assert cache == expected
    assert cache.is_dir()

    monkeypatch.chdir(tmp_path)
    ensure_numba_cache_dir()
    assert Path(os.environ["NUMBA_CACHE_DIR"]) == cache
