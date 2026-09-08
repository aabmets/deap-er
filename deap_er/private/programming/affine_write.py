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

from typing import Any

from .columnar import Window
from .compile_cache import expression_key
from .compilers import invalidate_compiled
from .primitives.primitive_nodes import Ephemeral
from .slim.slim_tree import SlimTree

__all__: list[str] = ["write_affine_scale"]

_ADD_NAMES = ("add", "vadd")
_MUL_NAMES = ("mul", "vmul")


class AffineEphemeral(Ephemeral):
    """Numeric leaf written by Lamarckian affine scaling."""

    ret = object

    @staticmethod
    def func() -> float:
        """Return a placeholder sampled before the fitted value is written."""
        return 0.0


def write_affine_scale(
    individual: Any,
    intercept: float,
    slope: float,
    prim_set: Any,
) -> Any:
    """Write Keijzer ``a + b * f(x)`` back onto a GP expression.

    A ``PrimitiveTree`` is wrapped as ``add(a, mul(b, tree))`` with
    ``a`` and ``b`` as ephemeral leaves. A ``SlimTree`` wraps the head
    the same way and wraps each existing delta as ``mul(b, delta)``,
    so compile stays ``a + b * (head + sum(deltas))``. Fitness and
    compile-cache entries for the previous expression are invalidated.
    This is the Lamarckian path next to
    :func:`~deap_er.gp.tune_ephemerals`; Darwinian callers use
    :func:`~deap_er.tools.affine_scale` only.

    Args:
        individual: ``PrimitiveTree`` or ``SlimTree`` to wrap in place.
        intercept: Fitted ``a``.
        slope: Fitted ``b``.
        prim_set: Primitive set that must provide ``add`` / ``vadd``
            and ``mul`` / ``vmul``.

    Returns:
        The same ``individual`` after write-back.

    Raises:
        TypeError: If ``add`` / ``mul`` (or the vector aliases) are
            missing from ``prim_set``.
    """
    add_prim = _require_primitive(prim_set, _ADD_NAMES, "addition")
    mul_prim = _require_primitive(prim_set, _MUL_NAMES, "multiplication")
    ret_type = _leaf_ret(add_prim)
    template = _ephemeral_template(prim_set, ret_type)
    old_keys = _expression_keys(individual)
    if isinstance(individual, SlimTree):
        _wrap_slim(individual, intercept, slope, add_prim, mul_prim, ret_type, template)
    else:
        _wrap_tree(individual, intercept, slope, add_prim, mul_prim, ret_type, template)
    for key in dict.fromkeys(old_keys):
        invalidate_compiled(key)
    _clear_fitness(individual)
    return individual


def _require_primitive(prim_set: Any, names: tuple[str, ...], role: str) -> Any:
    """Return the first registered primitive among ``names``."""
    mapping = getattr(prim_set, "mapping", {})
    for name in names:
        primitive = mapping.get(name)
        if primitive is not None:
            return primitive
    listed = ", ".join(repr(name) for name in names)
    raise TypeError(f"Affine writeback requires a {role} primitive ({listed}).")


def _leaf_ret(add_prim: Any) -> type:
    """Return the argument type used for written ``a`` and ``b`` leaves."""
    args = getattr(add_prim, "args", None)
    if args:
        return args[0]
    return object


def _ephemeral_template(prim_set: Any, ret_type: type) -> type[Ephemeral] | None:
    """Return a non-window ephemeral class from ``prim_set``, if any."""
    terminals = getattr(prim_set, "terminals", {})
    preferred = list(terminals.get(ret_type, []))
    others = [term for key, group in terminals.items() if key is not ret_type for term in group]
    for term in [*preferred, *others]:
        if _is_numeric_ephemeral_type(term):
            return term
    return None


def _is_numeric_ephemeral_type(term: Any) -> bool:
    """Return whether ``term`` is a float ephemeral class."""
    return (
        isinstance(term, type)
        and issubclass(term, Ephemeral)
        and getattr(term, "ret", None) is not Window
    )


def _scale_leaf(value: float, ret_type: type, template: type[Ephemeral] | None) -> Ephemeral:
    """Return an independent ephemeral carrying ``value``."""
    node = template() if template is not None else AffineEphemeral()
    node.ret = ret_type
    node.value = float(value)
    node.name = repr(float(value))
    return node


def _wrap_tree(
    tree: Any,
    intercept: float,
    slope: float,
    add_prim: Any,
    mul_prim: Any,
    ret_type: type,
    template: type[Ephemeral] | None,
) -> None:
    """Replace ``tree`` in place with ``add(a, mul(b, tree))``."""
    intercept_leaf = _scale_leaf(intercept, ret_type, template)
    slope_leaf = _scale_leaf(slope, ret_type, template)
    tree[:] = [add_prim, intercept_leaf, mul_prim, slope_leaf, *list(tree)]


def _wrap_slim(
    slim: SlimTree,
    intercept: float,
    slope: float,
    add_prim: Any,
    mul_prim: Any,
    ret_type: type,
    template: type[Ephemeral] | None,
) -> None:
    """Wrap the Slim head and each delta so compile is ``a + b * f``."""
    _wrap_tree(slim.head, intercept, slope, add_prim, mul_prim, ret_type, template)
    for delta in slim.deltas:
        slope_leaf = _scale_leaf(slope, ret_type, template)
        delta[:] = [mul_prim, slope_leaf, *list(delta)]


def _expression_keys(individual: Any) -> list[Any]:
    """Cache keys for the expression before write-back."""
    keys = [expression_key(individual)]
    if isinstance(individual, SlimTree):
        keys.append(expression_key(individual.head))
        keys.extend(expression_key(delta) for delta in individual.deltas)
    return keys


def _clear_fitness(individual: Any) -> None:
    """Drop valid fitness after the expression changes."""
    fitness = getattr(individual, "fitness", None)
    if fitness is not None and fitness.is_valid():
        del individual.fitness.values
