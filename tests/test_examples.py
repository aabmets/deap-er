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
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((REPO / "examples").glob("*/*.py"))
SCRIPT_IDS = [path.relative_to(REPO / "examples").as_posix() for path in SCRIPTS]


@pytest.mark.examples
@pytest.mark.parametrize("script", SCRIPTS, ids=SCRIPT_IDS)
def test_example_script(script: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
