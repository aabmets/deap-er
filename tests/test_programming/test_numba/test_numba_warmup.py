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
import subprocess
import sys
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


_WARM_PROBE = """
from deap_er import gp
from deap_er.private.programming.numba.numba_compile import build
gp.warmup_numba()
run, _ = build()
assert run.signatures, "the tape interpreter was not specialized"
"""


@pytest.mark.xdist_group(name="numba")
@pytest.mark.skipif(not gp.numba_available(), reason="the optional numba extra is not installed")
def test_warmup_numba_specializes_the_interpreter_in_a_fresh_process():
    completed = subprocess.run(
        [sys.executable, "-c", _WARM_PROBE], capture_output=True, text=True, timeout=600
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.xdist_group(name="numba")
@pytest.mark.skipif(not gp.numba_available(), reason="the optional numba extra is not installed")
def test_ensure_numba_cache_dir_reaches_an_already_imported_numba(tmp_path):
    # Optional extra: numba may be absent, in which case this test is skipped.
    import numba.core.config as config

    saved = {key: os.environ.get(key) for key in ("NUMBA_CACHE_DIR", "XDG_CACHE_HOME")}
    os.environ.pop("NUMBA_CACHE_DIR", None)
    os.environ["XDG_CACHE_HOME"] = str(tmp_path)
    try:
        ensure_numba_cache_dir()
        expected = (tmp_path / "deap-er" / "numba").resolve()
        assert Path(getattr(config, "CACHE_DIR")) == expected  # noqa: B009
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        config.reload_config()
