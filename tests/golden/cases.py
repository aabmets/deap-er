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
"""Seeded cases behind the golden-value tests.

Each ``*_case`` function returns the results of one seed as a JSON-friendly
mapping. The test modules compare those results against the stored data;
``_generate.py`` writes that data. Both go through this module, so a case is
defined exactly once.

Functions ending in ``_exact`` return values that are reproducible bit for
bit: comparisons, integer arithmetic, and the correctly rounded ``sqrt``.
Functions ending in ``_approx`` route through ``exp``, ``log``, ``sin``,
``cos``, fractional ``pow``, or LAPACK, any of which may differ by an ulp
between platforms, so their values are compared with a tolerance and are
returned flattened.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy
from deap_er import Fitness, creator, gp
from tests.golden.cases_approx import (
    benchmarks_case_approx,
    moving_peaks_case_approx,
    operators_case_approx,
    strategies_case_approx,
)
from tests.golden.cases_exact import (
    benchmarks_case_exact,
    ea_drivers_case_exact,
    gp_crossover_case_exact,
    operators_case_exact,
    selection_case_exact,
)

SEEDS = tuple(range(25))
DATA_DIR = Path(__file__).parent

_TYPE_NAMES = (
    "GOLDEN_FIT_MIN",
    "GOLDEN_IND_MIN",
    "GOLDEN_FIT_2",
    "GOLDEN_IND_2",
    "GOLDEN_FIT_4",
    "GOLDEN_IND_4",
    "GOLDEN_FIT_MO",
    "GOLDEN_IND_MO",
    "GOLDEN_FIT_GP",
    "GOLDEN_IND_GP",
)


def load(name: str) -> dict[str, Any]:
    """Read one stored golden data file.

    Args:
        name: File stem under ``tests/golden``.

    Returns:
        The stored results, keyed by seed.
    """
    return json.loads((DATA_DIR / f"{name}.json").read_text())


@contextmanager
def golden_types() -> Iterator[SimpleNamespace]:
    """Create the creator types the cases need, and remove them after.

    Yields:
        A namespace of the created individual classes.
    """
    creator.create_type("GOLDEN_FIT_MIN", Fitness, weights=(-1.0,))
    creator.create_type("GOLDEN_IND_MIN", list, fitness=creator.__dict__["GOLDEN_FIT_MIN"])
    creator.create_type("GOLDEN_FIT_2", Fitness, weights=(1.0, 1.0))
    creator.create_type("GOLDEN_IND_2", list, fitness=creator.__dict__["GOLDEN_FIT_2"])
    creator.create_type("GOLDEN_FIT_4", Fitness, weights=(1.0, 1.0, 1.0, 1.0))
    creator.create_type("GOLDEN_IND_4", list, fitness=creator.__dict__["GOLDEN_FIT_4"])
    creator.create_type("GOLDEN_FIT_MO", Fitness, weights=(-1.0, -1.0))
    creator.create_type("GOLDEN_IND_MO", numpy.ndarray, fitness=creator.__dict__["GOLDEN_FIT_MO"])
    creator.create_type("GOLDEN_FIT_GP", Fitness, weights=(-1.0,))
    creator.create_type(
        "GOLDEN_IND_GP", gp.PrimitiveTree, fitness=creator.__dict__["GOLDEN_FIT_GP"]
    )
    try:
        yield SimpleNamespace(
            min_ind=creator.__dict__["GOLDEN_IND_MIN"],
            two_obj=creator.__dict__["GOLDEN_IND_2"],
            four_obj=creator.__dict__["GOLDEN_IND_4"],
            mo_ind=creator.__dict__["GOLDEN_IND_MO"],
            gp_ind=creator.__dict__["GOLDEN_IND_GP"],
        )
    finally:
        for name in _TYPE_NAMES:
            del creator.__dict__[name]


CASES = {
    "selection_exact": selection_case_exact,
    "gp_crossover_exact": gp_crossover_case_exact,
    "ea_drivers_exact": ea_drivers_case_exact,
    "operators_exact": operators_case_exact,
    "operators_approx": operators_case_approx,
    "benchmarks_exact": benchmarks_case_exact,
    "benchmarks_approx": benchmarks_case_approx,
    "moving_peaks_approx": moving_peaks_case_approx,
    "strategies_approx": strategies_case_approx,
}
