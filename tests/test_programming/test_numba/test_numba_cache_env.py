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

import pytest
from deap_er import gp

# Numba's config is process-global, so the probe runs in a fresh
# interpreter that imports numba before the host sets the variable.
_LATE_ENV_PROBE = """
import os
import sys
os.environ.pop("NUMBA_CACHE_DIR", None)
import numba
os.environ["NUMBA_CACHE_DIR"] = sys.argv[1]
from deap_er.private.programming.numba.numba_compile import ensure_numba_cache_dir
ensure_numba_cache_dir()
from numba.core import config
print(config.CACHE_DIR)
"""


@pytest.mark.skipif(not gp.numba_available(), reason="the optional numba extra is not installed")
def test_ensure_numba_cache_dir_honours_a_variable_set_after_numba_import(tmp_path):
    target = str(tmp_path / "late-cache")
    env = {key: value for key, value in os.environ.items() if key != "NUMBA_CACHE_DIR"}

    completed = subprocess.run(
        [sys.executable, "-c", _LATE_ENV_PROBE, target],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == target
