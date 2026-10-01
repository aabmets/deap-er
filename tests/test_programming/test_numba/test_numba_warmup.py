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
from deap_er.private.programming.numba.numba_cache import ensure_numba_cache_dir


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
run = build()
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
def test_ensure_numba_cache_dir_reaches_an_already_imported_numba(tmp_path, monkeypatch):
    # Optional extra: numba may be absent, in which case this test is skipped.
    import numba.core.config as config

    try:
        with monkeypatch.context() as patch:
            # An empty value counts as unset and is restored on exit.
            patch.setenv("NUMBA_CACHE_DIR", "")
            patch.setenv("XDG_CACHE_HOME", str(tmp_path))
            ensure_numba_cache_dir()
            expected = (tmp_path / "deap-er" / "numba").resolve()
            assert Path(getattr(config, "CACHE_DIR")) == expected  # noqa: B009
    finally:
        config.reload_config()


_CACHE_PROBE = """
import numpy
from deap_er import gp
from deap_er.private.programming.numba.numba_batch import batch_kernel
from deap_er.private.programming.numba.numba_compile import build
pset = gp.make_column_pset(["first"])
gp.add_numpy_primitives(pset)
tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["first"]]), pset)
gp.bind_tape(tape)(numpy.zeros((2, 1)))
gp.interpret_tapes([tape], numpy.zeros((2, 1)), backend="numba")
for kernel in (build(), batch_kernel(False, False)):
    stats = kernel.stats
    print(sum(stats.cache_hits.values()), sum(stats.cache_misses.values()))
"""


def _interpreter_cache_files():
    cache = Path(os.environ["NUMBA_CACHE_DIR"])
    return sorted(path.name for path in cache.rglob("numba_kernels.interpret*"))


@pytest.mark.xdist_group(name="numba")
@pytest.mark.skipif(not gp.numba_available(), reason="the optional numba extra is not installed")
def test_a_fresh_process_loads_the_interpreter_from_the_disk_cache():
    # The interpreter used to take the dispatcher as an argument, whose
    # type differs in every process, so each one missed the cache and
    # appended another compiled copy to it. The batch kernel took the
    # interpreter as an argument and was never cached at all.
    gp.warmup_numba()
    before = _interpreter_cache_files()

    completed = subprocess.run(
        [sys.executable, "-c", _CACHE_PROBE], capture_output=True, text=True, timeout=600
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["1", "0", "1", "0"]
    assert _interpreter_cache_files() == before
