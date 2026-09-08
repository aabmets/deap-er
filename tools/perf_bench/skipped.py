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
"""Inventory features that are not timed as hot paths."""

SKIPPED_FEATURES = [
    {
        "feature": "tree_to_infix",
        "reason": "Display-only pretty printer; not timed evolutionary work.",
    },
    {
        "feature": "call_zero terminals",
        "reason": "Source-formatting flag for eval; not a computational hot path.",
    },
    {
        "feature": "empty Logbook header",
        "reason": "Print/stream cosmetic; not a runtime hot path.",
    },
    {
        "feature": "Logbook.to_json / from_json",
        "reason": "Serialization I/O, not an evolutionary hot path.",
    },
    {
        "feature": "ea_* logger",
        "reason": "Logging sink hook; log_time and fronts are benched instead.",
    },
]
