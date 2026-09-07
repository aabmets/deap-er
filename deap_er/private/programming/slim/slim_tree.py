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

from copy import deepcopy
from typing import Any, override

from ..compilers import compile_tree
from ..primitives.primitive_set_typed import PrimitiveSetTyped
from ..primitives.primitive_tree import PrimitiveTree

__all__: list[str] = ["SlimTree", "compile_slim_tree"]


class SlimTree:
    """SLIM geometric-semantic genotype: a head tree plus delta blocks.

    This type exists only to back the SLIM inflate, deflate, and donor
    operators. Standard GP should continue to use ``PrimitiveTree``.
    """

    fitness: Any

    def __init__(
        self,
        head: PrimitiveTree | list[Any],
        deltas: list[PrimitiveTree] | None = None,
    ) -> None:
        """Initialize a SLIM individual.

        Args:
            head: Initial tree ``T`` placed at the list head.
            deltas: Semantic delta blocks appended by inflate mutation.
        """
        self.head = head if isinstance(head, PrimitiveTree) else PrimitiveTree(head)
        self.deltas = [
            delta if isinstance(delta, PrimitiveTree) else PrimitiveTree(delta)
            for delta in (deltas or [])
        ]

    @classmethod
    def from_tree(cls, tree: SlimTree | PrimitiveTree | list[Any]) -> SlimTree:
        """Wrap a standard tree as a SLIM head with no deltas.

        Args:
            tree: Tree to use as the SLIM head.

        Returns:
            A new ``SlimTree`` when ``tree`` is not already one.
        """
        if isinstance(tree, SlimTree):
            return tree
        return cls(PrimitiveTree(tree))

    def __len__(self) -> int:
        """Return the total node count across head and delta blocks."""
        return len(self.head) + sum(len(delta) for delta in self.deltas)

    @override
    def __str__(self) -> str:
        """Return a Python expression that sums head and delta blocks."""
        parts = [f"({self.head})"] + [f"({delta})" for delta in self.deltas]
        return " + ".join(parts)

    def __deepcopy__(self, memo: dict[int, Any]) -> SlimTree:
        """Return a deep copy of this SLIM individual.

        Args:
            memo: Memo mapping used by ``copy.deepcopy``.

        Returns:
            A new ``SlimTree`` with copied head, deltas, and fitness.
        """
        clone = SlimTree(
            deepcopy(self.head, memo),
            [deepcopy(delta, memo) for delta in self.deltas],
        )
        if hasattr(self, "fitness"):
            clone.fitness = deepcopy(self.fitness, memo)
        return clone


def compile_slim_tree(
    slim: SlimTree,
    prim_set: PrimitiveSetTyped,
    *,
    backend: str = "python",
    dispatch: Any = None,
) -> Any:
    """Compile a SLIM individual for evaluation.

    Args:
        slim: SLIM genotype to compile.
        prim_set: Primitive set that supplies the evaluation context.
        backend: One of ``'python'``, ``'opcode'``, or ``'numba'``.
        dispatch: Compiled kernel for the ``'numba'`` backend.

    Returns:
        A callable if ``prim_set`` has one or more arguments,
        otherwise the result of the evaluation.
    """
    head_fn = compile_tree(slim.head, prim_set, backend=backend, dispatch=dispatch)
    if not slim.deltas:
        return head_fn
    delta_fns = [
        compile_tree(delta, prim_set, backend=backend, dispatch=dispatch) for delta in slim.deltas
    ]
    if len(prim_set.arguments) == 0:
        return head_fn() + sum(delta_fn() for delta_fn in delta_fns)

    def combined(*args: Any, **kwargs: Any) -> Any:
        total = head_fn(*args, **kwargs)
        for delta_fn in delta_fns:
            total += delta_fn(*args, **kwargs)
        return total

    return combined
