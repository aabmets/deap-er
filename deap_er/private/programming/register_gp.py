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
from operator import attrgetter
from typing import Any

from deap_er.private.operators.sel_tournament import sel_tournament
from deap_er.private.toolbox import Toolbox
from deap_er.private.various.clone import clone_individual
from deap_er.private.various.initializers import init_iterate, init_repeat

from .compilers import compile_tree
from .crossover import cx_one_point
from .generators import gen_full, gen_half_and_half
from .mutation import mut_uniform
from .primitives.primitive_set_typed import PrimitiveSetTyped
from .tree_graph import static_limit

__all__: list[str] = ["register_gp"]

_HEIGHT = attrgetter("height")


def register_gp(
    toolbox: Toolbox,
    pset: PrimitiveSetTyped,
    *,
    individual: type | None = None,
    min_depth: int = 1,
    max_depth: int = 2,
    mut_min_depth: int = 0,
    mut_max_depth: int = 2,
    height_limit: int | None = 17,
    backend: str | None = None,
    select: bool | Callable[..., Any] = True,
    contestants: int = 3,
) -> Toolbox:
    """Register the standard tree-GP operators on ``toolbox``.

    Wires the aliases every prefix-tree run needs and that callers
    commonly omit: ``clone_individual`` instead of ``deepcopy``,
    ``compile_tree``, half-and-half initialization, one-point
    crossover, uniform mutation, and a height ``static_limit``.
    Does not register ``evaluate`` or ``evaluate_batch`` — fitness
    stays on the caller.

    Args:
        toolbox: Toolbox to mutate.
        pset: Primitive set used for init, compile, and mutation.
        individual: Optional individual type. When given, registers
            ``expr``, ``individual``, and ``population``.
        min_depth: Minimum depth for half-and-half initialization.
        max_depth: Maximum depth for half-and-half initialization.
        mut_min_depth: Minimum depth of the subtree grown by mutation.
        mut_max_depth: Maximum depth of the subtree grown by mutation.
        height_limit: Height cap applied to ``mate`` and ``mutate``.
            ``None`` skips the decorator.
        backend: Optional ``compile_tree`` backend. The compile
            default is used when omitted.
        select: If True, register tournament selection. If False,
            leave ``select`` unset. A callable is registered as
            ``select`` instead.
        contestants: Tournament size when ``select`` is True.

    Returns:
        The same ``toolbox``, for chaining.
    """
    if individual is not None:
        toolbox.register(
            "expr",
            gen_half_and_half,
            prim_set=pset,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        toolbox.register("individual", init_iterate, individual, toolbox.expr)
        toolbox.register("population", init_repeat, list, toolbox.individual)
    toolbox.register("clone", clone_individual)
    if backend is None:
        toolbox.register("compile", compile_tree, prim_set=pset)
    else:
        toolbox.register("compile", compile_tree, prim_set=pset, backend=backend)
    toolbox.register("mate", cx_one_point)
    toolbox.register("expr_mut", gen_full, min_depth=mut_min_depth, max_depth=mut_max_depth)
    toolbox.register("mutate", mut_uniform, expr=toolbox.expr_mut, prim_set=pset)
    if height_limit is not None:
        limit = static_limit(_HEIGHT, height_limit)
        toolbox.decorate("mate", limit)
        toolbox.decorate("mutate", limit)
    if select is True:
        toolbox.register("select", sel_tournament, contestants=contestants)
    elif select is not False:
        toolbox.register("select", select)
    return toolbox
