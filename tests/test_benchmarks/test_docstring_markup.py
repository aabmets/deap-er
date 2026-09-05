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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIBRARY = _REPO_ROOT / "deap_er"

_FORBIDDEN = (":math:", ".. math::", ".. dropdown:: Equations", ".. list-table::")


def test_library_docstrings_do_not_use_rst_math_markup():
    hits: list[str] = []
    for path in sorted(_LIBRARY.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if any(marker in line for marker in _FORBIDDEN):
                rel = path.relative_to(_REPO_ROOT)
                hits.append(f"{rel}:{lineno}: {line.strip()}")
    assert hits == []
