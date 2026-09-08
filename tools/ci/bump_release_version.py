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
"""Set the release workflow default version from NEW_VERSION."""

import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release.yml"
_DEFAULT_RE = r'default: "[^"]+"  # gha-version-default'


def main() -> None:
    """Replace the marked ``default`` in ``release.yml`` with ``NEW_VERSION``."""
    new = os.environ.get("NEW_VERSION", "").strip()
    if not new:
        sys.exit("NEW_VERSION is not set")
    text = _WORKFLOW.read_text()
    updated, n = re.subn(
        _DEFAULT_RE,
        f'default: "{new}"  # gha-version-default',
        text,
        count=1,
    )
    if n != 1:
        sys.exit(f"Failed to update version default (replacements={n})")
    _WORKFLOW.write_text(updated)


if __name__ == "__main__":
    main()
