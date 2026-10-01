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

import pickle
from collections.abc import Callable
from typing import Any

from . import primitive_nodes
from .primitive_nodes import Ephemeral

__all__: list[str] = ["ephemeral_class", "registered_ephemeral", "restore_ephemeral"]

_RESERVED = frozenset([*primitive_nodes.__all__, "PrimitiveSetTyped"])
_portable: dict[type, bool] = {}


def ephemeral_class(name: str, func: Callable[..., Any], ret_type: type) -> type[Ephemeral]:
    """Return the ephemeral class named ``name``, creating it on first use.

    The class lives in this module under its own name, so that one name
    means one ephemeral across every primitive set of the process.

    Args:
        name: Name of the ephemeral type.
        func: Zero-arity callable that produces a value.
        ret_type: Type returned by the ephemeral.

    Returns:
        The ephemeral class.

    Raises:
        TypeError: If ``name`` is already used by a different ephemeral
            or by a class of the gp module.
    """
    module_gp = globals()
    if name not in module_gp and name not in _RESERVED:
        attrs: dict[str, Any] = {"func": staticmethod(func), "ret": ret_type}
        attrs |= {"__module__": __name__, "__qualname__": name, "__reduce__": _reduce}
        class_ = type(name, (Ephemeral,), attrs)
        module_gp[name] = class_
        return class_

    class_ = module_gp.get(name)
    if not (isinstance(class_, type) and issubclass(class_, Ephemeral)) or name in _RESERVED:
        raise TypeError(
            "Ephemera should be named differently than classes defined in the gp module."
        )
    if class_.func is not func:
        raise TypeError(
            "Ephemera with different functions should be named differently even between psets."
        )
    if class_.ret is not ret_type:
        raise TypeError(
            "Ephemera with the same name and function should have the same type even between psets."
        )
    return class_


def registered_ephemeral(name: str, module: str) -> type[Ephemeral]:
    """Return the registered ephemeral class ``name`` for ``module``.

    Raises:
        AttributeError: If no ephemeral of that name is registered.
    """
    class_ = globals().get(name)
    if isinstance(class_, type) and issubclass(class_, Ephemeral) and class_ is not Ephemeral:
        return class_
    raise AttributeError(f"module {module!r} has no attribute {name!r}")


def restore_ephemeral(name: str, ret_type: type, func: Callable[..., Any] | None) -> Ephemeral:
    """Return a blank instance of ephemeral ``name`` for unpickling.

    An ephemeral already registered in this process is reused. Otherwise
    the class is rebuilt from ``func``, so a tree loads in a fresh
    process before any primitive set has registered its ephemerals.

    Args:
        name: Name of the ephemeral type.
        ret_type: Type returned by the ephemeral.
        func: The pickled sampler, or None if it could not be pickled.

    Returns:
        An uninitialized instance; pickle restores its value.

    Raises:
        pickle.UnpicklingError: If ``name`` is not registered and
            ``func`` is None.
    """
    try:
        class_ = registered_ephemeral(name, __name__)
    except AttributeError as error:
        if func is None:
            raise pickle.UnpicklingError(
                f"The ephemeral '{name}' is not registered in this process and its "
                f"function could not be pickled. Register it on a primitive set "
                f"before loading."
            ) from error
        class_ = ephemeral_class(name, func, ret_type)
    return class_.__new__(class_)


def _reduce(node: Ephemeral) -> tuple[Any, ...]:
    """Pickle an ephemeral node with what is needed to rebuild its class."""
    class_ = type(node)
    func = class_.func if _is_portable(class_) else None
    return restore_ephemeral, (class_.__name__, class_.ret, func), node.__getstate__()


def _is_portable(class_: type[Ephemeral]) -> bool:
    """Return whether the sampler of ``class_`` can travel to another process."""
    if class_ not in _portable:
        _portable[class_] = _can_pickle(class_.func)
    return _portable[class_]


def _can_pickle(func: Callable[..., Any]) -> bool:
    """Return whether ``func`` pickles for another process.

    A sampler defined in ``__main__`` is left out, because another
    process may run a different main module. A tree whose sampler is
    left out loads only where its ephemeral is already registered.
    """
    if getattr(func, "__module__", None) == "__main__":
        return False
    try:
        pickle.dumps(func)
    except (pickle.PicklingError, AttributeError, TypeError):
        return False
    return True
