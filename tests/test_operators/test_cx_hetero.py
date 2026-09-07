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
from functools import partial
from typing import Any

import numpy
import pytest
from deap_er import Toolbox, tools


class _FitList(list[Any]):
    def __init__(self, genes: list[Any], values: tuple[float, ...] = (1.0,)) -> None:
        super().__init__(genes)
        self.fitness = type("Fit", (), {"values": values})()


def _swap(left: Any, right: Any) -> tuple[Any, Any]:
    return right, left


def test_per_gene_applies_one_callable_and_is_inplace():
    first: Any = [1, 2, 3]
    second: Any = [10, 20, 30]
    mates = tools.cx_heterogeneous(
        first, second, [lambda a, b: (a + 1, b), _swap, lambda a, b: (0, 0)]
    )
    assert mates == ([2, 20, 0], [10, 2, 0])
    assert mates[0] is first
    assert mates[1] is second


def test_per_slice_uniform_and_blend_write_back():
    first: Any = [0, 1, 2, 0.1, 0.2]
    second: Any = [9, 8, 7, 0.9, 0.8]
    tools.rng.seed(0)
    tools.cx_heterogeneous(
        first,
        second,
        (
            (slice(0, 3), partial(tools.cx_uniform, cx_prob=1.0)),
            (slice(3, None), partial(tools.cx_blend_bounded, alpha=0.5, low=0.0, up=1.0)),
        ),
    )
    assert first[:3] == [9, 8, 7]
    assert second[:3] == [0, 1, 2]
    assert all(0.0 <= gene <= 1.0 for gene in first[3:] + second[3:])
    assert first[3:] != [0.1, 0.2] or second[3:] != [0.9, 0.8]


def test_open_tail_and_gap_leave_uncovered_genes():
    first: Any = [1, 2, 3, 4, 5]
    second: Any = [9, 8, 7, 6, 5]
    tools.cx_heterogeneous(
        first,
        second,
        ((slice(0, 1), _as_seq_swap), (slice(3, None), _as_seq_swap)),
    )
    assert first == [9, 2, 3, 6, 5]
    assert second == [1, 8, 7, 4, 5]


def _as_seq_swap(left: Any, right: Any) -> tuple[Any, Any]:
    return list(right), list(left)


def test_toolbox_register_partial_slice_specs():
    toolbox = Toolbox()
    toolbox.register(
        "mate",
        tools.cx_heterogeneous,
        crossovers=(
            (slice(0, 2), partial(tools.cx_uniform, cx_prob=1.0)),
            (slice(2, None), partial(tools.cx_blend_bounded, alpha=0.5, low=0.0, up=1.0)),
        ),
    )
    first: Any = [0, 1, 0.2, 0.3]
    second: Any = [8, 9, 0.8, 0.7]
    tools.rng.seed(1)
    mates = toolbox.mate(first, second)
    assert mates[0] is first
    assert first[:2] == [8, 9]
    assert second[:2] == [0, 1]


def test_aligned_length_mismatch_raises():
    first: Any = [1, 2]
    second: Any = [3, 4]
    with pytest.raises(ValueError, match="same length"):
        tools.cx_heterogeneous(first, second, [_swap])


def test_unequal_individuals_raise():
    first: Any = [1, 2]
    second: Any = [3]
    with pytest.raises(ValueError, match="individuals must have the same length"):
        tools.cx_heterogeneous(first, second, [_swap, _swap])


def test_overlapping_slices_raise():
    first: Any = [1, 2, 3]
    second: Any = [4, 5, 6]
    with pytest.raises(ValueError, match="must not overlap"):
        tools.cx_heterogeneous(
            first,
            second,
            ((slice(0, 2), _as_seq_swap), (slice(1, 3), _as_seq_swap)),
        )


def test_stepped_slice_raises():
    first: Any = [1, 2, 3, 4]
    second: Any = [5, 6, 7, 8]
    with pytest.raises(ValueError, match="contiguous"):
        tools.cx_heterogeneous(first, second, ((slice(0, 4, 2), _as_seq_swap),))


def test_mixed_shapes_raise():
    first: Any = [1, 2]
    second: Any = [3, 4]
    with pytest.raises(ValueError, match="either one callable per gene"):
        tools.cx_heterogeneous(first, second, [_swap, (slice(1, 2), _as_seq_swap)])
    with pytest.raises(ValueError, match="either one callable per gene"):
        tools.cx_heterogeneous(first, second, ((slice(0, 1), _as_seq_swap), _swap))


def test_non_callable_raises():
    first: Any = [1]
    second: Any = [2]
    with pytest.raises(ValueError, match="must be callable"):
        tools.cx_heterogeneous(first, second, [None])
    with pytest.raises(ValueError, match="must be callable"):
        tools.cx_heterogeneous(first, second, ((slice(0, 1), None),))


def test_length_changing_slice_operator_raises():
    def grow(left: Any, right: Any) -> tuple[Any, Any]:
        return [*left, 0], [*right, 0]

    first: Any = [1, 2]
    second: Any = [3, 4]
    with pytest.raises(ValueError, match="preserve unit length"):
        tools.cx_heterogeneous(first, second, ((slice(0, 2), grow),))


def test_empty_individuals_with_empty_crossovers_are_noop():
    first: Any = []
    second: Any = []
    mates = tools.cx_heterogeneous(first, second, [])
    assert mates == ([], [])
    assert mates[0] is first


def test_numpy_slice_writeback_does_not_alias_parents():
    first: Any = numpy.array([0, 1, 2, 3], dtype=int)
    second: Any = numpy.array([9, 8, 7, 6], dtype=int)
    tools.rng.seed(2)
    tools.cx_heterogeneous(first, second, ((slice(0, 4), partial(tools.cx_uniform, cx_prob=1.0)),))
    assert list(first) == [9, 8, 7, 6]
    assert list(second) == [0, 1, 2, 3]
    first[0] = -1
    assert second[0] == 0


def test_fitness_values_are_left_intact():
    first: Any = _FitList([1, 2], (4.0,))
    second: Any = _FitList([3, 4], (5.0,))
    tools.cx_heterogeneous(first, second, [_swap, _swap])
    assert first.fitness.values == (4.0,)
    assert second.fitness.values == (5.0,)


def test_uniform_slice_matches_per_index_random_stream():
    left_genes = [0, 1, 2, 3]
    right_genes = [9, 8, 7, 6]
    tools.rng.seed(11)
    draws = [tools.rng.random() for _ in range(len(left_genes))]
    expected_left = [
        right_genes[i] if draw < 0.35 else left_genes[i] for i, draw in enumerate(draws)
    ]
    expected_right = [
        left_genes[i] if draw < 0.35 else right_genes[i] for i, draw in enumerate(draws)
    ]
    tools.rng.seed(11)
    first: Any = list(left_genes)
    second: Any = list(right_genes)
    tools.cx_heterogeneous(first, second, ((slice(0, 4), partial(tools.cx_uniform, cx_prob=0.35)),))
    assert first == expected_left
    assert second == expected_right


def test_per_gene_bad_return_raises():
    first: Any = [1]
    second: Any = [2]
    with pytest.raises(ValueError, match="two replacements"):
        tools.cx_heterogeneous(first, second, [lambda a, b: a])


def test_flat_tuple_slice_unit_dispatches_shape_b():
    first: Any = [1, 2, 3]
    second: Any = [9, 8, 7]
    tools.cx_heterogeneous(first, second, (slice(0, 3), _as_seq_swap))
    assert first == [9, 8, 7]
    assert second == [1, 2, 3]


def test_flat_list_slice_unit_dispatches_shape_b():
    first: Any = [1, 2]
    second: Any = [9, 8]
    tools.cx_heterogeneous(first, second, [slice(0, 2), _as_seq_swap])
    assert first == [9, 8]
    assert second == [1, 2]


def test_nested_list_pair_units():
    first: Any = [1, 2, 3]
    second: Any = [9, 8, 7]
    tools.cx_heterogeneous(
        first,
        second,
        [[slice(0, 1), _as_seq_swap], [slice(2, 3), _as_seq_swap]],
    )
    assert first == [9, 2, 7]
    assert second == [1, 8, 3]


def test_inverted_slice_raises_clear_error():
    first: Any = [1, 2, 3, 4, 5]
    second: Any = [9, 8, 7, 6, 5]
    with pytest.raises(ValueError, match="inverted|start"):
        tools.cx_heterogeneous(first, second, ((slice(4, 2), _as_seq_swap),))
