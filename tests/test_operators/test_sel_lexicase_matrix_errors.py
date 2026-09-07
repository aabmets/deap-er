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
import numpy
import pytest
from deap_er import tools
from deap_er.private.operators.sel_lexicase_matrix import (
    fitness_case_matrix,
    lexicase_select_vectorized,
    validate_case_matrix,
)


def test_fitness_case_matrix_empty_and_mismatched(multi_obj, make):
    with pytest.raises(ValueError, match="non-empty"):
        fitness_case_matrix([])

    first = make(multi_obj, [0], (1.0, 2.0))
    second = make(multi_obj, [1], (3.0, 4.0))

    class _Short:
        values = (3.0,)

    second.fitness = _Short()
    with pytest.raises(ValueError, match="same length"):
        fitness_case_matrix([first, second])


def test_validate_case_matrix_empty_and_length_mismatch(multi_obj, make):
    with pytest.raises(ValueError, match="non-empty"):
        validate_case_matrix(numpy.zeros((0, 0)), [])

    first = make(multi_obj, [0], (1.0, 2.0))
    second = make(multi_obj, [1], (3.0, 4.0))
    matrix = tools.fitness_case_matrix([first, second])

    class _Short:
        values = (3.0,)

    second.fitness = _Short()
    with pytest.raises(ValueError, match="same length"):
        validate_case_matrix(matrix, [first, second])


def test_vectorized_epsilon_minimize_and_empty_survivors(multi_obj, make):
    creator_fit = make(multi_obj, [0], (1.0, 0.0))
    other = make(multi_obj, [1], (0.0, 1.0))
    population = [creator_fit, other]
    matrix = numpy.array([[numpy.nan, numpy.nan], [numpy.nan, numpy.nan]])

    tools.rng.seed(0)
    chosen = lexicase_select_vectorized(
        population,
        3,
        matrix,
        [0],
        (-1.0, -1.0),
        mode="epsilon_auto",
    )
    assert len(chosen) == 3
    assert all(ind in population for ind in chosen)

    with pytest.raises(ValueError, match="epsilon must be set"):
        lexicase_select_vectorized(
            population,
            1,
            tools.fitness_case_matrix(population),
            [0],
            (1.0, 1.0),
            mode="epsilon_fixed",
        )
