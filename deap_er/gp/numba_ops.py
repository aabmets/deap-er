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
import sys
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
_UNKNOWN_OPCODE = "The tape holds an opcode the interpreter does not know."

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


def _truthy(value: float) -> bool:  # pragma: no cover
    """Treat a 0/1 mask cell as a boolean without an equality test.

    Args:
        value: Stack cell written by a comparison or logic opcode.

    Returns:
        True when ``value`` is away from zero.
    """
    return abs(value) > 0.0


def _apply_numeric(  # pragma: no cover
    op: int,
    rows: int,
    sp: int,
    stack: Any,
    columns: Any,
    constants: Any,
    arg: int,
    fill: float,
) -> int:
    """Apply a load or arithmetic opcode.

    Args:
        op: Opcode in the numeric group.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        columns: Packed input matrix.
        constants: Constant pool.
        arg: Immediate operand.
        fill: Protected-op fill.

    Returns:
        The updated stack pointer.

    Raises:
        ValueError: If ``op`` is not a numeric opcode.
    """
    if op in (_COL_LOAD, _CONST):
        return _apply_load(op, rows, sp, stack, columns, constants, arg)
    if op <= _DIV:
        return _apply_arith(op, rows, sp, stack, fill)
    if op in (_NEG, _ABS, _SIN, _COS):
        return _apply_unary(op, rows, sp, stack)
    if op == _LOG:
        return _apply_log(rows, sp, stack, fill)
    if op == _SQRT:
        return _apply_sqrt(rows, sp, stack, fill)
    raise ValueError(_UNKNOWN_OPCODE)


def _apply_load(  # pragma: no cover
    op: int,
    rows: int,
    sp: int,
    stack: Any,
    columns: Any,
    constants: Any,
    arg: int,
) -> int:
    """Push a column or a constant onto the stack.

    Args:
        op: ``COL_LOAD`` or ``CONST``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        columns: Packed input matrix.
        constants: Constant pool.
        arg: Column index or constant-pool index.

    Returns:
        The updated stack pointer.
    """
    if op == _COL_LOAD:
        for t in range(rows):
            stack[sp, t] = columns[t, arg]
        return sp + 1
    value = constants[arg]
    for t in range(rows):
        stack[sp, t] = value
    return sp + 1


def _apply_arith(op: int, rows: int, sp: int, stack: Any, fill: float) -> int:  # pragma: no cover
    """Apply a binary arithmetic opcode.

    Args:
        op: ``ADD``, ``SUB``, ``MUL``, or ``DIV``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        fill: Protected-op fill.

    Returns:
        The updated stack pointer.

    Raises:
        ValueError: If ``op`` is not a binary arithmetic opcode.
    """
    if op == _ADD:
        sp -= 1
        for t in range(rows):
            stack[sp - 1, t] = stack[sp - 1, t] + stack[sp, t]
        return sp
    if op == _SUB:
        sp -= 1
        for t in range(rows):
            stack[sp - 1, t] = stack[sp - 1, t] - stack[sp, t]
        return sp
    if op == _MUL:
        sp -= 1
        for t in range(rows):
            stack[sp - 1, t] = stack[sp - 1, t] * stack[sp, t]
        return sp
    if op == _DIV:
        return _apply_div(rows, sp, stack, fill)
    raise ValueError(_UNKNOWN_OPCODE)


def _apply_unary(op: int, rows: int, sp: int, stack: Any) -> int:  # pragma: no cover
    """Apply a unary numeric opcode.

    Args:
        op: ``NEG``, ``ABS``, ``SIN``, or ``COS``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.

    Returns:
        The updated stack pointer.

    Raises:
        ValueError: If ``op`` is not a unary numeric opcode.
    """
    if op == _NEG:
        for t in range(rows):
            stack[sp - 1, t] = -stack[sp - 1, t]
        return sp
    if op == _ABS:
        for t in range(rows):
            stack[sp - 1, t] = abs(stack[sp - 1, t])
        return sp
    if op == _SIN:
        for t in range(rows):
            stack[sp - 1, t] = math.sin(stack[sp - 1, t])
        return sp
    if op == _COS:
        for t in range(rows):
            stack[sp - 1, t] = math.cos(stack[sp - 1, t])
        return sp
    raise ValueError(_UNKNOWN_OPCODE)


def _apply_div(rows: int, sp: int, stack: Any, fill: float) -> int:  # pragma: no cover
    """Apply protected division.

    Args:
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        fill: Value used when finite operands produce a non-finite result.

    Returns:
        The updated stack pointer.
    """
    sp -= 1
    for t in range(rows):
        left = stack[sp - 1, t]
        right = stack[sp, t]
        value = left / right
        if not math.isfinite(value) and math.isfinite(left) and math.isfinite(right):
            value = fill
        stack[sp - 1, t] = value
    return sp


def _apply_log(rows: int, sp: int, stack: Any, fill: float) -> int:  # pragma: no cover
    """Apply the protected natural logarithm.

    Args:
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        fill: Value used for a non-positive finite operand.

    Returns:
        The updated stack pointer.
    """
    for t in range(rows):
        value = stack[sp - 1, t]
        if value > 0.0:
            stack[sp - 1, t] = math.log(value)
        elif math.isfinite(value):
            stack[sp - 1, t] = fill
        else:
            stack[sp - 1, t] = math.nan
    return sp


def _apply_sqrt(rows: int, sp: int, stack: Any, fill: float) -> int:  # pragma: no cover
    """Apply the protected square root.

    Args:
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        fill: Value used for a negative finite operand.

    Returns:
        The updated stack pointer.
    """
    for t in range(rows):
        value = stack[sp - 1, t]
        if value >= 0.0:
            stack[sp - 1, t] = math.sqrt(value)
        elif math.isfinite(value):
            stack[sp - 1, t] = fill
        else:
            stack[sp - 1, t] = math.nan
    return sp


def _apply_predicate(op: int, rows: int, sp: int, stack: Any) -> int:  # pragma: no cover
    """Apply a comparison, logic, or selection opcode.

    Args:
        op: Opcode in the predicate group.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.

    Returns:
        The updated stack pointer.

    Raises:
        ValueError: If ``op`` is not a predicate opcode.
    """
    if op in (_GT, _LT):
        return _apply_gt_lt(op, rows, sp, stack)
    if op in (_GE, _LE):
        return _apply_ge_le(op, rows, sp, stack)
    if op == _EQ:
        return _apply_eq(rows, sp, stack)
    if op in (_AND, _OR):
        return _apply_and_or(op, rows, sp, stack)
    if op <= _WHERE:
        return _apply_not_where(op, rows, sp, stack)
    raise ValueError(_UNKNOWN_OPCODE)


def _apply_gt_lt(op: int, rows: int, sp: int, stack: Any) -> int:  # pragma: no cover
    """Apply a strict comparison opcode.

    Args:
        op: ``GT`` or ``LT``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.

    Returns:
        The updated stack pointer.
    """
    sp -= 1
    if op == _GT:
        for t in range(rows):
            stack[sp - 1, t] = 1.0 if stack[sp - 1, t] > stack[sp, t] else 0.0
        return sp
    for t in range(rows):
        stack[sp - 1, t] = 1.0 if stack[sp - 1, t] < stack[sp, t] else 0.0
    return sp


def _apply_ge_le(op: int, rows: int, sp: int, stack: Any) -> int:  # pragma: no cover
    """Apply an inclusive comparison opcode.

    Args:
        op: ``GE`` or ``LE``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.

    Returns:
        The updated stack pointer.
    """
    sp -= 1
    if op == _GE:
        for t in range(rows):
            stack[sp - 1, t] = 1.0 if stack[sp - 1, t] >= stack[sp, t] else 0.0
        return sp
    for t in range(rows):
        stack[sp - 1, t] = 1.0 if stack[sp - 1, t] <= stack[sp, t] else 0.0
    return sp


def _apply_eq(rows: int, sp: int, stack: Any) -> int:  # pragma: no cover
    """Apply exact equality without a floating-point ``==``.

    Both inequalities are True only when the values are equal and
    finite, matching ``numpy.equal``.

    Args:
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.

    Returns:
        The updated stack pointer.
    """
    sp -= 1
    for t in range(rows):
        left = stack[sp - 1, t]
        right = stack[sp, t]
        stack[sp - 1, t] = 1.0 if left <= right and left >= right else 0.0
    return sp


def _apply_and_or(op: int, rows: int, sp: int, stack: Any) -> int:  # pragma: no cover
    """Apply a binary logic opcode.

    Args:
        op: ``AND`` or ``OR``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.

    Returns:
        The updated stack pointer.
    """
    sp -= 1
    if op == _AND:
        for t in range(rows):
            stack[sp - 1, t] = 1.0 if _truthy(stack[sp - 1, t]) and _truthy(stack[sp, t]) else 0.0
        return sp
    for t in range(rows):
        stack[sp - 1, t] = 1.0 if _truthy(stack[sp - 1, t]) or _truthy(stack[sp, t]) else 0.0
    return sp


def _apply_not_where(op: int, rows: int, sp: int, stack: Any) -> int:  # pragma: no cover
    """Apply ``NOT`` or ``WHERE``.

    Args:
        op: ``NOT`` or ``WHERE``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.

    Returns:
        The updated stack pointer.
    """
    if op == _NOT:
        for t in range(rows):
            stack[sp - 1, t] = 0.0 if _truthy(stack[sp - 1, t]) else 1.0
        return sp
    sp -= 2
    for t in range(rows):
        stack[sp - 1, t] = stack[sp, t] if _truthy(stack[sp - 1, t]) else stack[sp + 1, t]
    return sp


def _apply_window(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> int:
    """Apply a delay, difference, rolling, or EMA opcode.

    Args:
        op: Opcode in the window group.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Window length.

    Returns:
        The updated stack pointer.

    Raises:
        ValueError: If ``op`` is not a window opcode.
    """
    if op in (_DELAY, _DIFF):
        _apply_shift(op, rows, sp, stack, arg)
        return sp
    if op in (_ROLL_SUM, _ROLL_MEAN, _ROLL_STD):
        _roll_stats(op, rows, sp, stack, scratch, arg)
        return sp
    if op in (_ROLL_MIN, _ROLL_MAX):
        _roll_minmax(op, rows, sp, stack, scratch, arg)
        return sp
    if op == _EMA:
        _roll_ema(rows, sp, stack, scratch, arg)
        return sp
    raise ValueError(_UNKNOWN_OPCODE)


def _apply_shift(op: int, rows: int, sp: int, stack: Any, arg: int) -> None:  # pragma: no cover
    """Apply a causal delay or difference.

    Args:
        op: ``DELAY`` or ``DIFF``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        arg: Shift length.
    """
    if op == _DELAY:
        for t in range(rows - 1, -1, -1):
            stack[sp - 1, t] = stack[sp - 1, t - arg] if t >= arg else math.nan
    else:
        for t in range(rows - 1, -1, -1):
            if t >= arg:
                stack[sp - 1, t] = stack[sp - 1, t] - stack[sp - 1, t - arg]
            else:
                stack[sp - 1, t] = math.nan


def _roll_stats(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> None:
    """Write a rolling sum, mean, or population standard deviation.

    Args:
        op: One of the rolling reduction opcodes.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Window length.
    """
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
        scratch[t] = _reduce_stats(op, total, squares, arg)
    for t in range(rows):
        stack[sp - 1, t] = scratch[t]


def _reduce_stats(op: int, total: float, squares: float, arg: int) -> float:  # pragma: no cover
    """Reduce one full window to a sum, mean, or standard deviation.

    Args:
        op: Rolling opcode.
        total: Sum of the window.
        squares: Sum of squares of the window.
        arg: Window length.

    Returns:
        The reduced value.
    """
    if op == _ROLL_SUM:
        return total
    if op == _ROLL_MEAN:
        return total / arg
    mean = total / arg
    variance = squares / arg - mean * mean
    if variance < 0.0:
        variance = 0.0
    return math.sqrt(variance)


def _roll_minmax(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> None:
    """Write a rolling minimum or maximum.

    Args:
        op: ``ROLL_MIN`` or ``ROLL_MAX``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Window length.
    """
    for t in range(rows):
        if t + 1 < arg:
            scratch[t] = math.nan
            continue
        scratch[t] = _window_extreme(op, stack, sp, t - arg + 1, t + 1)
    for t in range(rows):
        stack[sp - 1, t] = scratch[t]


def _window_extreme(  # pragma: no cover
    op: int, stack: Any, sp: int, begin: int, end: int
) -> float:
    """Return the min or max of ``stack[sp - 1, begin:end]``.

    A ``nan`` in the window makes the result ``nan``.

    Args:
        op: ``ROLL_MIN`` or ``ROLL_MAX``.
        stack: Column-length workspace.
        sp: Current stack pointer.
        begin: Inclusive start index.
        end: Exclusive stop index.

    Returns:
        The extreme value, or ``nan``.
    """
    best = stack[sp - 1, begin]
    for j in range(begin + 1, end):
        value = stack[sp - 1, j]
        if math.isnan(value) or math.isnan(best):
            best = math.nan
        elif op == _ROLL_MIN:
            if value < best:
                best = value
        elif value > best:
            best = value
    return float(best)


def _roll_ema(rows: int, sp: int, stack: Any, scratch: Any, arg: int) -> None:  # pragma: no cover
    """Write a causal exponential moving average.

    Args:
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Span of the average.
    """
    start = -1
    for t in range(rows):
        if math.isfinite(stack[sp - 1, t]):
            start = t
            break
    if start < 0 or arg > rows:
        for t in range(rows):
            scratch[t] = math.nan
    else:
        _fill_ema(rows, sp, stack, scratch, arg, start)
    for t in range(rows):
        stack[sp - 1, t] = scratch[t]


def _fill_ema(  # pragma: no cover
    rows: int, sp: int, stack: Any, scratch: Any, arg: int, start: int
) -> None:
    """Fill ``scratch`` with the EMA of ``stack[sp - 1]`` from ``start``.

    Args:
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Span of the average.
        start: Index of the first finite sample.
    """
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


def _interpret(  # pragma: no cover
    opcodes: Any,
    operands: Any,
    constants: Any,
    columns: Any,
    fill: float,
    stack: Any,
    scratch: Any,
    dispatch: Any,
) -> int:
    """Run one tape through the compiled stack machine.

    Args:
        opcodes: Instruction stream.
        operands: Immediate operand of each instruction.
        constants: Constant pool.
        columns: Packed input matrix.
        fill: Protected-op fill.
        stack: Column-length workspace.
        scratch: Spare row.
        dispatch: Consumer kernel.

    Returns:
        The stack pointer after the last instruction.

    Raises:
        ValueError: If an opcode is unknown.
    """
    rows = columns.shape[0]
    sp = 0
    for step in range(opcodes.size):
        op = opcodes[step]
        arg = operands[step]
        if op <= _COS:
            sp = _apply_numeric(op, rows, sp, stack, columns, constants, arg, fill)
        elif op <= _WHERE:
            sp = _apply_predicate(op, rows, sp, stack)
        elif op <= _EMA:
            sp = _apply_window(op, rows, sp, stack, scratch, arg)
        elif op >= _BASE:
            sp = int(dispatch(op, sp, stack, columns, constants, scratch))
        else:
            raise ValueError(_UNKNOWN_OPCODE)
    return sp


def _idle(  # pragma: no cover
    _op: int,
    _sp: int,
    _stack: Any,
    _columns: Any,
    _constants: Any,
    _scratch: Any,
) -> int:
    """Reject an unexpected consumer opcode.

    The parameter names are unused; they exist so the fallback kernel
    matches ``USER_DISPATCH_SIGNATURE``.

    Returns:
        ``-1``, which is not a valid stack pointer.
    """
    return -1


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
    # Callees first so the interpreter sees compiled globals. Numba
    # compiles these from bytecode, so the CPython tracer never sees
    # them. The parity tests exercise every instruction.
    module = sys.modules[__name__]
    for name in (
        "_truthy",
        "_apply_div",
        "_apply_log",
        "_apply_sqrt",
        "_apply_load",
        "_apply_arith",
        "_apply_unary",
        "_apply_eq",
        "_apply_gt_lt",
        "_apply_ge_le",
        "_apply_and_or",
        "_apply_not_where",
        "_apply_shift",
        "_reduce_stats",
        "_window_extreme",
        "_fill_ema",
        "_roll_stats",
        "_roll_minmax",
        "_roll_ema",
        "_apply_numeric",
        "_apply_predicate",
        "_apply_window",
        "_interpret",
        "_idle",
    ):
        setattr(module, name, jit(getattr(module, name)))
    _built["run"] = module._interpret
    _built["idle"] = module._idle
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
