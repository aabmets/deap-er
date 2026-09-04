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
"""Regenerate the stored golden values.

Run from the repository root after an intentional behaviour change::

    uv run python -m tests.golden._generate

Review the resulting diff before committing: every changed value is a change
in library behaviour.
"""

import json

from tests.golden.cases import CASES, DATA_DIR, SEEDS, golden_types


def main() -> None:
    """Write one JSON file per case family, one line per seed."""
    with golden_types() as types:
        for name, case in CASES.items():
            lines = [f'"{seed}": {json.dumps(case(types, seed), sort_keys=True)}' for seed in SEEDS]
            path = DATA_DIR / f"{name}.json"
            path.write_text("{\n" + ",\n".join(lines) + "\n}\n")
            print(f"wrote {path.relative_to(DATA_DIR.parents[1])} ({len(lines)} seeds)")


if __name__ == "__main__":
    main()
