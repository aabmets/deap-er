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
"""Compatibility shim for the hot-path chart.

Prefer ``uv run python -m tools.perf_bench`` (writes JSON then the chart)
or ``from tools.perf_bench import write_chart``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.perf_bench.cli import plot_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(plot_main())
