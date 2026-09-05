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
from collections.abc import Callable
from functools import partial
from typing import Any

from ._numpy_arith import (
    DEFAULT_FILL,
    vabs,
    vadd,
    vcos,
    vdiv,
    vlog,
    vmul,
    vneg,
    vsin,
    vsqrt,
    vsub,
)
from ._numpy_logic import vand, veq, vge, vgt, vle, vlt, vnot, vor, vwhere
from .columnar import Array, Mask, _reject_shadowed
from .primitives import PrimitiveSetTyped

__all__ = [
    "vadd",
    "vsub",
    "vmul",
    "vneg",
    "vabs",
    "vdiv",
    "vlog",
    "vsqrt",
    "vsin",
    "vcos",
    "vgt",
    "vlt",
    "vge",
    "vle",
    "veq",
    "vand",
    "vor",
    "vnot",
    "vwhere",
    "add_numpy_primitives",
    "infer_fill",
]

_PROTECTED_NAMES = ("vdiv", "vlog", "vsqrt")


def infer_fill(prim_set: PrimitiveSetTyped) -> float:
    """Recover the protection fill that a primitive set was built with.

    Reads the value bound into the protected primitives by
    ``add_numpy_primitives``, so an alternative backend reproduces the
    same protection without being told the value again.

    Args:
        prim_set: Primitive set to inspect.

    Returns:
        The bound fill, or ``DEFAULT_FILL`` when the set carries no
        protected primitive.
    """
    for name in _PROTECTED_NAMES:
        func = prim_set.context.get(name)
        if isinstance(func, partial):
            value = func.keywords.get("fill")
            if value is not None:
                return float(value)
    return DEFAULT_FILL


def add_numpy_primitives(prim_set: PrimitiveSetTyped, *, fill: float = DEFAULT_FILL) -> None:
    """Register the vectorized primitive kit on a typed primitive set.

    Adds arithmetic, protected division and domain-limited unary
    operations, comparisons that return ``Mask``, mask logic, and
    ``vwhere``. The constants ``True`` and ``False`` are registered as
    ``Mask`` terminals so tree generation can terminate a mask branch.

    The chosen ``fill`` is bound into the protected primitives, where
    ``infer_fill`` can read it back so the other backends reproduce the
    same protection.

    Args:
        prim_set: Typed primitive set built by ``make_column_pset`` or
            an equivalent set over ``Array`` and ``Mask``.
        fill: Value substituted for a non-finite result of a protected
            operation that was computed from finite operands.

    Raises:
        ValueError: If a primitive name collides with an argument of
            ``prim_set``, or if a name is already registered.
    """
    binary: list[type] = [Array, Array]
    unary: list[type] = [Array]

    plain: dict[str, tuple[Callable[..., Any], list[type]]] = {
        "vadd": (vadd, binary),
        "vsub": (vsub, binary),
        "vmul": (vmul, binary),
        "vneg": (vneg, unary),
        "vabs": (vabs, unary),
        "vsin": (vsin, unary),
        "vcos": (vcos, unary),
    }
    protected: dict[str, tuple[Callable[..., Any], list[type]]] = {
        "vdiv": (vdiv, binary),
        "vlog": (vlog, unary),
        "vsqrt": (vsqrt, unary),
    }
    compares: dict[str, Callable[..., Any]] = {
        "vgt": vgt,
        "vlt": vlt,
        "vge": vge,
        "vle": vle,
        "veq": veq,
    }
    logic: dict[str, tuple[Callable[..., Any], list[type]]] = {
        "vand": (vand, [Mask, Mask]),
        "vor": (vor, [Mask, Mask]),
        "vnot": (vnot, [Mask]),
    }

    names = [*plain, *protected, *compares, *logic, "vwhere"]
    _reject_shadowed(prim_set, names)

    for name, (func, in_types) in plain.items():
        prim_set.add_primitive(func, in_types, Array, name)
    for name, (func, in_types) in protected.items():
        prim_set.add_primitive(partial(func, fill=fill), in_types, Array, name)
    for name, func in compares.items():
        prim_set.add_primitive(func, binary, Mask, name)
    for name, (func, in_types) in logic.items():
        prim_set.add_primitive(func, in_types, Mask, name)

    prim_set.add_primitive(vwhere, [Mask, Array, Array], Array, "vwhere")
    prim_set.add_terminal(True, Mask)
    prim_set.add_terminal(False, Mask)
