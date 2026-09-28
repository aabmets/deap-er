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
from __future__ import annotations

from collections.abc import Callable

from .policy_push_ops import (
    LOAD_COVERAGE,
    LOAD_HELD_OUT,
    LOAD_HELD_OUT_SET,
    LOAD_INVALID,
    LOAD_NE_EVALS,
    LOAD_PROMOTED,
    LOAD_QD,
    LOAD_REJECTED,
    LOAD_ROWS,
    LOAD_SOLVE_BIT,
    LOAD_TRAIN_SCORE,
    LOAD_UNSOLVED,
    PUSH_BOOL,
    PUSH_INT,
    PushInterpState,
    read_operand,
)

__all__: list[str] = ["PUSH_HANDLERS"]


def _push_int(state: PushInterpState) -> None:
    state.stack.append(read_operand(state))


def _push_bool(state: PushInterpState) -> None:
    state.stack.append(bool(read_operand(state)))


def _load_unsolved(state: PushInterpState) -> None:
    state.stack.append(state.observation.unsolved_count)


def _load_nevals(state: PushInterpState) -> None:
    state.stack.append(state.observation.nevals)


def _load_rows(state: PushInterpState) -> None:
    state.stack.append(state.observation.rows_seen)


def _load_promoted(state: PushInterpState) -> None:
    state.stack.append(state.observation.promoted_library_size)


def _load_invalid(state: PushInterpState) -> None:
    state.stack.append(state.observation.fitness_invalid)


def _load_rejected(state: PushInterpState) -> None:
    state.stack.append(state.observation.last_action_rejected)


def _load_train_score(state: PushInterpState) -> None:
    state.stack.append(state.observation.train_score)


def _load_held_out(state: PushInterpState) -> None:
    if state.observation.held_out_score is None:
        raise ValueError("held_out_score is not available")
    state.stack.append(state.observation.held_out_score)


def _load_held_out_set(state: PushInterpState) -> None:
    state.stack.append(state.observation.held_out_score is not None)


def _load_coverage(state: PushInterpState) -> None:
    state.stack.append(state.observation.archive_coverage)


def _load_qd(state: PushInterpState) -> None:
    state.stack.append(state.observation.qd_score)


def _load_solve_bit(state: PushInterpState) -> None:
    bit_index = read_operand(state)
    bits = state.observation.solve_bits
    value = bits[bit_index] if 0 <= bit_index < len(bits) else 0
    state.stack.append(value)


PUSH_HANDLERS: dict[int, Callable[[PushInterpState], None]] = {
    PUSH_INT: _push_int,
    PUSH_BOOL: _push_bool,
    LOAD_UNSOLVED: _load_unsolved,
    LOAD_NE_EVALS: _load_nevals,
    LOAD_ROWS: _load_rows,
    LOAD_PROMOTED: _load_promoted,
    LOAD_INVALID: _load_invalid,
    LOAD_REJECTED: _load_rejected,
    LOAD_TRAIN_SCORE: _load_train_score,
    LOAD_HELD_OUT: _load_held_out,
    LOAD_HELD_OUT_SET: _load_held_out_set,
    LOAD_COVERAGE: _load_coverage,
    LOAD_QD: _load_qd,
    LOAD_SOLVE_BIT: _load_solve_bit,
}
