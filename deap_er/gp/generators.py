#
#   MIT License
#   
#   Copyright (c) 2022, Mattias Aabmets
#   
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#   
#   SPDX-License-Identifier: MIT
#
from .primitives import PrimitiveSetTyped
from typing import Callable, Optional, Any
from inspect import isclass
import random
import sys


__all__ = ['generate', 'gen_full', 'gen_grow', 'gen_half_and_half']


# ====================================================================================== #
def generate(prim_set: PrimitiveSetTyped, min_depth: int,
             max_depth: int, condition: Callable,
             ret_type: Optional[Any] = None) -> list:
    """
    Generates a tree as a list of primitives and terminals in a depth-first order.
    The tree is built from the root to the leaves. It recursively grows each branch
    until the **condition** is fulfilled. The returned list can then be used to
    instantiate a 'PrimitiveTree' object to build the actual tree object.

    :param prim_set: Primitive set from which primitives are selected.
    :param min_depth: Minimum depth of the random tree.
    :param max_depth: Maximum depth of the random tree.
    :param condition: A function that takes two arguments: the height
        of the branch to grow and the current depth in the tree.
    :param ret_type: The type that should return the tree when called,
        optional. If not provided, the type of 'p_set.ret' is used.
    :return: A grown tree with leaves at possibly different
        depths depending on the condition function.
    """
    err_msg = "The gp.generate function tried to add a {0} " \
              "of type \'{1}\', but there is none available."
    if ret_type is None:
        ret_type = prim_set.ret
    expr = list()
    height = random.randint(min_depth, max_depth)
    stack = [(0, ret_type)]
    while len(stack) != 0:
        depth, ret_type = stack.pop()
        if condition(height, depth):
            try:
                term = random.choice(prim_set.terminals[ret_type])
                if isclass(term):
                    term = term()
                expr.append(term)
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    err_msg.format('terminal', ret_type)
                ).with_traceback(traceback)
        else:
            try:
                prim = prim_set.primitives[ret_type]
                prim = random.choice(prim)
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    err_msg.format('primitive', ret_type)
                ).with_traceback(traceback)
    return expr


# -------------------------------------------------------------------------------------- #
def gen_full(prim_set: PrimitiveSetTyped, min_depth: int,
             max_depth: int, ret_type: Optional[Any] = None) -> list:
    """
    Generates an expression where each leaf has the same
    depth between **min** and **max**.

    :param prim_set: Primitive set from which primitives are selected.
    :param min_depth: Minimum depth of the random tree.
    :param max_depth: Maximum depth of the random tree.
    :param ret_type: The type that should return the tree when called,
        optional. If not provided, the type of 'p_set.ret' is used.
    :return: A full tree with all leaves at the same depth.
    """
    def condition(height, depth):
        return height == depth
    return generate(prim_set, min_depth, max_depth, condition, ret_type)


# -------------------------------------------------------------------------------------- #
def gen_grow(prim_set: PrimitiveSetTyped, min_depth: int,
             max_depth: int, ret_type: Optional[Any] = None) -> list:
    """
    Generates an expression where each leaf might have a different
    depth between **min** and **max**.

    :param prim_set: Primitive set from which primitives are selected.
    :param min_depth: Minimum depth of the random tree.
    :param max_depth: Maximum depth of the random tree.
    :param ret_type: The type that should return the tree when called,
        optional. If not provided, the type of 'p_set.ret' is used.
    :return: A grown tree with leaves at possibly different depths.
    """
    def condition(height, depth):
        cond = random.random() < prim_set.terminal_ratio
        return depth == height or (depth >= min_depth and cond)
    return generate(prim_set, min_depth, max_depth, condition, ret_type)


# -------------------------------------------------------------------------------------- #
def gen_half_and_half(prim_set: PrimitiveSetTyped, min_depth: int,
                      max_depth: int, ret_type: Optional[Any] = None) -> list:
    """
    Generates an expression with a random choice
    between *'gen_grow'* and *'gen_full'*.

    :param prim_set: Primitive set from which primitives are selected.
    :param min_depth: Minimum depth of the random tree.
    :param max_depth: Maximum depth of the random tree.
    :param ret_type: The type that should return the tree when called,
        optional. If not provided, the type of 'p_set.ret' is used.
    :return: Either a full tree or a grown tree.
    """
    choices = (gen_grow, gen_full)
    func = random.choice(choices)
    return func(prim_set, min_depth, max_depth, ret_type)
