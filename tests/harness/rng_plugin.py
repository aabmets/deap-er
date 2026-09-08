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
import random

import numpy
import pytest
from deap_er import tools

__all__ = ["TEST_SEED", "seed_test_rng", "pytest_configure", "deterministic_rng"]

TEST_SEED = 0


def seed_test_rng(seed: int = TEST_SEED) -> None:
    """Reseed the process-wide generators used by library code and tests.

    Args:
        seed: Value passed to ``tools.rng``, ``random``, and NumPy.
    """
    tools.rng.seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def pytest_configure(config: pytest.Config) -> None:
    seed_test_rng()


@pytest.fixture(autouse=True)
def deterministic_rng() -> None:
    seed_test_rng()
