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
"""Require NEW_VERSION to be a PEP 440 version greater than pyproject.toml."""

import os
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def _project_version() -> str:
    """Return the ``version`` field from the repo ``pyproject.toml``.

    Returns:
        The current project version string.

    Raises:
        SystemExit: If the version field is missing.
    """
    for line in _PYPROJECT.read_text().splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit(f"Could not read project.version from {_PYPROJECT}")


def main() -> None:
    """Compare ``NEW_VERSION`` to the project version and exit on failure."""
    raw = os.environ.get("NEW_VERSION", "").strip()
    if not raw:
        sys.exit("NEW_VERSION is not set")
    current = _project_version()
    try:
        new = Version(raw)
        cur = Version(current)
    except InvalidVersion as exc:
        sys.exit(f"Invalid version: {exc}")
    if new <= cur:
        sys.exit(f"Version {raw} must be greater than current {current}")
    print(f"{raw} > {current}")


if __name__ == "__main__":
    main()
