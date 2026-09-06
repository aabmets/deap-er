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
import keyword
import re
from collections.abc import Sequence

from .primitives.primitive_set_typed import PrimitiveSetTyped

__all__: list[str] = [
    "Array",
    "Mask",
    "Window",
    "reject_shadowed",
    "validate_names",
    "make_column_pset",
]

_ARG_PATTERN = re.compile(r"^ARG\d+$")


class Array:
    """Type tag for a one-dimensional ``float64`` column.

    Used as a strongly typed genetic programming type. Values that
    flow through nodes tagged with it are NumPy arrays, never
    instances of this class.
    """


class Mask:
    """Type tag for a one-dimensional boolean condition.

    Comparison and logic primitives return this tag. It keeps
    conditions out of arithmetic positions and gives ``where`` a
    well-defined first argument.
    """


class Window:
    """Type tag for a positive integer window length.

    Rolling and delay primitives take one argument of this tag. No
    primitive returns it, so a window is always a leaf and can be
    lowered to an immediate operand.
    """


def reject_shadowed(prim_set: PrimitiveSetTyped, names: list[str]) -> None:
    """Reject primitive names that a compiled lambda parameter shadows.

    Args:
        prim_set: Primitive set the names are about to be added to.
        names: Primitive names about to be registered.

    Raises:
        ValueError: If a name is also an argument of ``prim_set``.
    """
    arguments = set(prim_set.arguments)
    for name in names:
        if name in arguments:
            raise ValueError(
                f"Primitive name '{name}' is also an argument of the primitive set. "
                f"A compiled lambda parameter would shadow the primitive."
            )


def validate_names(names: Sequence[str]) -> None:
    """Reject column names that cannot be used as lambda parameters.

    Args:
        names: Column names to check.

    Raises:
        ValueError: If ``names`` is empty, holds a duplicate, holds a
            name that is not a plain identifier, or holds a name that
            collides with the default argument prefix.
    """
    if len(names) == 0:
        raise ValueError("At least one column name is required.")

    seen: set[str] = set()
    for name in names:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"Column name '{name}' is not a valid Python identifier.")
        if keyword.iskeyword(name) or keyword.issoftkeyword(name):
            raise ValueError(f"Column name '{name}' is a Python keyword.")
        if _ARG_PATTERN.match(name):
            raise ValueError(
                f"Column name '{name}' collides with the default argument "
                f"prefix and would alias another column during renaming."
            )
        if name in seen:
            raise ValueError(f"Column name '{name}' is not unique.")
        seen.add(name)


def make_column_pset(names: Sequence[str], name: str = "MAIN") -> PrimitiveSetTyped:
    """Build a typed primitive set with one ``Array`` input per column.

    The resulting set expects and returns ``Array``. Argument order is
    the column order: ``pset.arguments[i]`` is ``names[i]``, and a
    callable from ``compile_tree`` takes the columns as positional
    arguments in that same order.

    Register operators onto the set with ``add_numpy_primitives``,
    ``add_window_primitives``, ``add_pair_window_primitives``, and
    ``add_ts_primitives``.

    Args:
        names: Column names, in the order the columns are passed to
            compiled trees. Each name must be a plain Python
            identifier that is not a keyword and does not look like
            the default ``ARG<n>`` argument prefix.
        name: Name of the primitive set.

    Returns:
        A typed primitive set with one renamed input terminal per
        column.

    Raises:
        ValueError: If ``names`` is empty or holds an unusable name.
    """
    validate_names(names)
    in_types: list[type] = [Array] * len(names)
    prim_set = PrimitiveSetTyped(name, in_types, Array)
    prim_set.rename_arguments(**{f"ARG{i}": column for i, column in enumerate(names)})
    return prim_set
