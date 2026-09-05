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
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy

from .opcodes import USER_BASE, Opcode, Tape

__all__ = ["USER_DISPATCH_SIGNATURE", "bind_tape", "numba_available"]

USER_DISPATCH_SIGNATURE = (
    "(op: int64, sp: int64, stack: float64[:, ::1], columns: float64[:, ::1], "
    "constants: float64[::1], scratch: float64[::1]) -> int64"
)
"""Signature every consumer dispatch kernel must have.

The interpreter keeps one stack of column-length rows. ``sp`` is the
number of occupied rows, so the top operand is ``stack[sp - 1]`` and a
kernel of arity ``n`` reads ``stack[sp - n]`` through ``stack[sp - 1]``,
writes its result into ``stack[sp - n]``, and returns ``sp - n + 1``::

    stack[sp - 1]  <- top operand      (n_rows values)
    stack[sp - 2]  <- second operand
    ...
    stack[sp - n]  <- first operand, and where the result goes
    stack[sp]      <- free scratch row

``constants`` is the tape constant pool, ``scratch`` is one spare row
of ``n_rows`` values that a kernel may clobber freely, and ``stack[sp]``
is always allocated and free to use as well. Builtin rolling primitives
fold their window length into the instruction, but a consumer primitive
receives every argument on the stack, so a window arrives as a
broadcast row and can be read from ``stack[sp - 1][0]``.
"""

_COL_LOAD = int(Opcode.COL_LOAD)
_CONST = int(Opcode.CONST)
_ADD = int(Opcode.ADD)
_SUB = int(Opcode.SUB)
_MUL = int(Opcode.MUL)
_DIV = int(Opcode.DIV)
_NEG = int(Opcode.NEG)
_ABS = int(Opcode.ABS)
_LOG = int(Opcode.LOG)
_SQRT = int(Opcode.SQRT)
_SIN = int(Opcode.SIN)
_COS = int(Opcode.COS)
_GT = int(Opcode.GT)
_LT = int(Opcode.LT)
_GE = int(Opcode.GE)
_LE = int(Opcode.LE)
_EQ = int(Opcode.EQ)
_AND = int(Opcode.AND)
_OR = int(Opcode.OR)
_NOT = int(Opcode.NOT)
_WHERE = int(Opcode.WHERE)
_DELAY = int(Opcode.DELAY)
_DIFF = int(Opcode.DIFF)
_ROLL_SUM = int(Opcode.ROLL_SUM)
_ROLL_MEAN = int(Opcode.ROLL_MEAN)
_ROLL_STD = int(Opcode.ROLL_STD)
_ROLL_MIN = int(Opcode.ROLL_MIN)
_ROLL_MAX = int(Opcode.ROLL_MAX)
_EMA = int(Opcode.EMA)
_BASE = USER_BASE

_MISSING = (
    "The numba backend needs the optional 'numba' dependency. "
    "Install it with: pip install deap-er[numba]"
)

_built: dict[str, Any] = {}
_workspace: dict[str, numpy.ndarray] = {}


def _reserve(depth: int, rows: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Hand out the process-wide interpreter workspace.

    One workspace is shared by every compiled tape so that a population
    of cached programs cannot pin one large buffer each. It grows to
    fit the deepest tape seen so far and is replaced when the number of
    rows changes. Sharing it is what makes the compiled callables
    unsafe to run from several threads at once.

    Args:
        depth: Peak stack depth of the tape.
        rows: Number of samples per column.

    Returns:
        A stack of ``depth + 1`` rows, and one scratch row. The extra
        row is the one ``USER_DISPATCH_SIGNATURE`` promises consumer
        kernels at ``stack[sp]``.
    """
    stack = _workspace.get("stack")
    if stack is None or stack.shape[1] != rows or stack.shape[0] < depth + 1:
        stack = numpy.empty((depth + 1, rows), dtype=numpy.float64)
        _workspace["stack"] = stack
        _workspace["scratch"] = numpy.empty(rows, dtype=numpy.float64)
    return stack[: depth + 1], _workspace["scratch"]


def numba_available() -> bool:
    """Report whether the optional Numba dependency is importable.

    Returns:
        True when the ``numba`` extra is installed.
    """
    try:
        import numba  # noqa: F401  # optional extra, probed without requiring it
    except ImportError:
        return False
    return True


def _build() -> tuple[Any, Any]:
    """Compile the tape interpreter and the fallback dispatcher.

    Both are built once per process and reused for every tape.

    Returns:
        The interpreter and the no-op dispatcher.

    Raises:
        ImportError: If the ``numba`` extra is not installed.
    """
    if "run" in _built:
        return _built["run"], _built["idle"]
    try:
        import numba  # optional extra, imported only when the backend is asked for
    except ImportError as err:
        raise ImportError(_MISSING) from err

    jit = numba.njit(cache=False, nogil=True, error_model="numpy")

    # Numba compiles the two functions below from bytecode, so the
    # interpreter never runs under the CPython tracer and coverage
    # cannot see it. The parity tests exercise every instruction.

    def idle(  # pragma: no cover
        op: int,
        sp: int,
        stack: Any,
        columns: Any,
        constants: Any,
        scratch: Any,
    ) -> int:
        return -1

    def run(  # pragma: no cover
        opcodes: Any,
        operands: Any,
        constants: Any,
        columns: Any,
        fill: float,
        stack: Any,
        scratch: Any,
        dispatch: Any,
    ) -> int:
        rows = columns.shape[0]
        sp = 0
        for step in range(opcodes.size):
            op = opcodes[step]
            arg = operands[step]

            if op == _COL_LOAD:
                for t in range(rows):
                    stack[sp, t] = columns[t, arg]
                sp += 1
            elif op == _CONST:
                value = constants[arg]
                for t in range(rows):
                    stack[sp, t] = value
                sp += 1
            elif op == _ADD:
                sp -= 1
                for t in range(rows):
                    stack[sp - 1, t] = stack[sp - 1, t] + stack[sp, t]
            elif op == _SUB:
                sp -= 1
                for t in range(rows):
                    stack[sp - 1, t] = stack[sp - 1, t] - stack[sp, t]
            elif op == _MUL:
                sp -= 1
                for t in range(rows):
                    stack[sp - 1, t] = stack[sp - 1, t] * stack[sp, t]
            elif op == _DIV:
                sp -= 1
                for t in range(rows):
                    left = stack[sp - 1, t]
                    right = stack[sp, t]
                    value = left / right
                    if not math.isfinite(value) and math.isfinite(left) and math.isfinite(right):
                        value = fill
                    stack[sp - 1, t] = value
            elif op == _NEG:
                for t in range(rows):
                    stack[sp - 1, t] = -stack[sp - 1, t]
            elif op == _ABS:
                for t in range(rows):
                    stack[sp - 1, t] = abs(stack[sp - 1, t])
            elif op == _SIN:
                for t in range(rows):
                    stack[sp - 1, t] = math.sin(stack[sp - 1, t])
            elif op == _COS:
                for t in range(rows):
                    stack[sp - 1, t] = math.cos(stack[sp - 1, t])
            elif op == _LOG:
                for t in range(rows):
                    value = stack[sp - 1, t]
                    if value > 0.0:
                        stack[sp - 1, t] = math.log(value)
                    elif math.isfinite(value):
                        stack[sp - 1, t] = fill
                    else:
                        stack[sp - 1, t] = math.nan
            elif op == _SQRT:
                for t in range(rows):
                    value = stack[sp - 1, t]
                    if value >= 0.0:
                        stack[sp - 1, t] = math.sqrt(value)
                    elif math.isfinite(value):
                        stack[sp - 1, t] = fill
                    else:
                        stack[sp - 1, t] = math.nan
            elif op == _GT:
                sp -= 1
                for t in range(rows):
                    stack[sp - 1, t] = 1.0 if stack[sp - 1, t] > stack[sp, t] else 0.0
            elif op == _LT:
                sp -= 1
                for t in range(rows):
                    stack[sp - 1, t] = 1.0 if stack[sp - 1, t] < stack[sp, t] else 0.0
            elif op == _GE:
                sp -= 1
                for t in range(rows):
                    stack[sp - 1, t] = 1.0 if stack[sp - 1, t] >= stack[sp, t] else 0.0
            elif op == _LE:
                sp -= 1
                for t in range(rows):
                    stack[sp - 1, t] = 1.0 if stack[sp - 1, t] <= stack[sp, t] else 0.0
            elif op == _EQ:
                sp -= 1
                for t in range(rows):
                    stack[sp - 1, t] = 1.0 if stack[sp - 1, t] == stack[sp, t] else 0.0
            elif op == _AND:
                sp -= 1
                for t in range(rows):
                    hit = stack[sp - 1, t] != 0.0 and stack[sp, t] != 0.0
                    stack[sp - 1, t] = 1.0 if hit else 0.0
            elif op == _OR:
                sp -= 1
                for t in range(rows):
                    hit = stack[sp - 1, t] != 0.0 or stack[sp, t] != 0.0
                    stack[sp - 1, t] = 1.0 if hit else 0.0
            elif op == _NOT:
                for t in range(rows):
                    stack[sp - 1, t] = 1.0 if stack[sp - 1, t] == 0.0 else 0.0
            elif op == _WHERE:
                sp -= 2
                for t in range(rows):
                    keep = stack[sp - 1, t] != 0.0
                    stack[sp - 1, t] = stack[sp, t] if keep else stack[sp + 1, t]
            elif op == _DELAY:
                for t in range(rows - 1, -1, -1):
                    stack[sp - 1, t] = stack[sp - 1, t - arg] if t >= arg else math.nan
            elif op == _DIFF:
                for t in range(rows - 1, -1, -1):
                    if t >= arg:
                        stack[sp - 1, t] = stack[sp - 1, t] - stack[sp - 1, t - arg]
                    else:
                        stack[sp - 1, t] = math.nan
            elif op in (_ROLL_SUM, _ROLL_MEAN, _ROLL_STD):
                for t in range(rows):
                    if t + 1 < arg:
                        scratch[t] = math.nan
                        continue
                    total = 0.0
                    squares = 0.0
                    for j in range(t - arg + 1, t + 1):
                        value = stack[sp - 1, j]
                        total += value
                        squares += value * value
                    if op == _ROLL_SUM:
                        scratch[t] = total
                    elif op == _ROLL_MEAN:
                        scratch[t] = total / arg
                    else:
                        mean = total / arg
                        variance = squares / arg - mean * mean
                        if variance < 0.0:
                            variance = 0.0
                        scratch[t] = math.sqrt(variance)
                for t in range(rows):
                    stack[sp - 1, t] = scratch[t]
            elif op in (_ROLL_MIN, _ROLL_MAX):
                for t in range(rows):
                    if t + 1 < arg:
                        scratch[t] = math.nan
                        continue
                    best = stack[sp - 1, t - arg + 1]
                    for j in range(t - arg + 2, t + 1):
                        value = stack[sp - 1, j]
                        if math.isnan(value) or math.isnan(best):
                            best = math.nan
                        elif op == _ROLL_MIN:
                            if value < best:
                                best = value
                        else:
                            if value > best:
                                best = value
                    scratch[t] = best
                for t in range(rows):
                    stack[sp - 1, t] = scratch[t]
            elif op == _EMA:
                start = -1
                for t in range(rows):
                    if math.isfinite(stack[sp - 1, t]):
                        start = t
                        break
                if start < 0 or arg > rows:
                    for t in range(rows):
                        scratch[t] = math.nan
                else:
                    alpha = 2.0 / (arg + 1.0)
                    previous = stack[sp - 1, start]
                    for t in range(start):
                        scratch[t] = math.nan
                    scratch[start] = previous
                    for t in range(start + 1, rows):
                        previous = alpha * stack[sp - 1, t] + (1.0 - alpha) * previous
                        scratch[t] = previous
                    stop = start + arg - 1
                    if stop > rows:
                        stop = rows
                    for t in range(stop):
                        scratch[t] = math.nan
                for t in range(rows):
                    stack[sp - 1, t] = scratch[t]
            elif op >= _BASE:
                sp = int(dispatch(op, sp, stack, columns, constants, scratch))
            else:
                raise ValueError("The tape holds an opcode the interpreter does not know.")
        return sp

    _built["run"] = jit(run)
    _built["idle"] = jit(idle)
    return _built["run"], _built["idle"]


def _as_matrix(columns: Sequence[Any], expected: int) -> numpy.ndarray:
    """Pack the arguments of a compiled tree into one contiguous matrix.

    Args:
        columns: Either one two-dimensional matrix, or one array per
            column.
        expected: Number of columns the tape expects.

    Returns:
        A C-contiguous ``(n_rows, n_columns)`` ``float64`` matrix.

    Raises:
        ValueError: If the column count does not match the tape.
    """
    if len(columns) == 1 and numpy.ndim(columns[0]) == 2:
        matrix = numpy.ascontiguousarray(columns[0], dtype=numpy.float64)
    else:
        if len(columns) != expected:
            raise ValueError(f"The tape expects {expected} columns, got {len(columns)}.")
        parts = [numpy.asarray(column, dtype=numpy.float64) for column in columns]
        matrix = numpy.ascontiguousarray(numpy.stack(parts, axis=1))
    if matrix.shape[1] != expected:
        raise ValueError(f"The tape expects {expected} columns, got {matrix.shape[1]}.")
    return matrix


def bind_tape(tape: Tape, dispatch: Any = None) -> Callable[..., numpy.ndarray]:
    """Bind a tape to the compiled interpreter.

    The interpreter is compiled once per process and once more for each
    distinct dispatcher, never per tree. It is single threaded on
    purpose: parallelize across individuals through ``toolbox.map``
    rather than inside a tree.

    Every compiled tape shares one process-wide workspace, so the
    returned callable must not be run from several threads at once. The
    result it returns is a fresh array.

    Args:
        tape: Tape produced by ``lower_tree``.
        dispatch: Compiled kernel implementing the opcodes at or above
            ``USER_BASE``, following ``USER_DISPATCH_SIGNATURE``.

    Returns:
        A callable that takes one array per column, or a single
        ``(n_rows, n_columns)`` matrix, and returns the result.

    Raises:
        ImportError: If the ``numba`` extra is not installed.
        ValueError: If the tape expects no columns, or if it holds
            consumer opcodes but no dispatcher was given.
    """
    if tape.columns == 0:
        raise ValueError(
            "The numba backend evaluates a tape over columns and cannot size a "
            "result without them. Use backend='python' or backend='opcode' for a "
            "primitive set that takes no arguments."
        )
    run, idle = _build()
    if dispatch is None:
        unknown = tape.opcodes[tape.opcodes >= USER_BASE]
        if unknown.size:
            raise ValueError(
                f"The tape holds consumer opcode {int(unknown[0])} but no dispatch "
                f"kernel was given. Pass dispatch= to compile_tree."
            )
        dispatch = idle

    def call(*columns: Any) -> numpy.ndarray:
        matrix = _as_matrix(columns, tape.columns)
        stack, scratch = _reserve(tape.depth, matrix.shape[0])
        run(
            tape.opcodes,
            tape.operands,
            tape.constants,
            matrix,
            tape.fill,
            stack,
            scratch,
            dispatch,
        )
        return stack[0].copy()

    return call
