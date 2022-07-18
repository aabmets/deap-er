# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 - Mattias Aabmets, The DEAP Team and Other Contributors           #
#                                                                                        #
#   Permission is hereby granted, free of charge, to any person obtaining a copy         #
#   of this software and associated documentation files (the "Software"), to deal        #
#   in the Software without restriction, including without limitation the rights         #
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell            #
#   copies of the Software, and to permit persons to whom the Software is                #
#   furnished to do so, subject to the following conditions:                             #
#                                                                                        #
#   The above copyright notice and this permission notice shall be included in all       #
#   copies or substantial portions of the Software.                                      #
#                                                                                        #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR           #
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,             #
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE          #
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER               #
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,        #
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE        #
#   SOFTWARE.                                                                            #
#                                                                                        #
# ====================================================================================== #
from .primitives import PrimitiveSetTyped
from typing import Callable
from inspect import isclass
import random
import sys


__all__ = ['generate', 'gen_full', 'gen_grow', 'gen_half_and_half']


# ====================================================================================== #
def generate(p_set: PrimitiveSetTyped, min_depth: int,
             max_depth: int, condition: Callable, ret_type=None) -> list:
    """
    Generates a tree as a list of primitives and terminals in a depth-first order.
    The tree is built from the root to the leaves. It recursively grows each branch
    until the *condition* is fulfilled. The returned list can then be used to
    instantiate a *PrimitiveTree* object to build the actual tree object.

    Parameters:
        p_set: Primitive set from which primitives are selected.
        min_depth: Minimum depth of the random tree.
        max_depth: Maximum depth of the random tree.
        condition: A function that takes two arguments: the height
            of the branch to grow and the current depth in the tree.
        ret_type: The type that should return the tree when called.
            If None, the type of pset.ret is used.
    Returns:
        A grown tree with leaves at possibly different
        depths depending on the condition function.
    """
    err_msg = "The gp.generate function tried to add a {0} " \
              "of type \'{1}\', but there is none available."
    if ret_type is None:
        ret_type = p_set.ret
    expr = list()
    height = random.randint(min_depth, max_depth)
    stack = [(0, ret_type)]
    while len(stack) != 0:
        depth, ret_type = stack.pop()
        if condition(height, depth):
            try:
                term = random.choice(p_set.terminals[ret_type])
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
                prim = random.choice(p_set.primitives[ret_type])
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
def gen_full(p_set: PrimitiveSetTyped, min_depth: int,
             max_depth: int, ret_type=None) -> list:
    """
    Generates an expression where each leaf has the same
    depth between *min* and *max*.

    Parameters:
        p_set: Primitive set from which primitives are selected.
        min_depth: Minimum depth of the random tree.
        max_depth: Maximum depth of the random tree.
        ret_type: The type that should return the tree when called.
            If None, the type of pset.ret is used.
    Returns:
        A full tree with all leaves at the same depth.
    """
    def condition(height, depth):
        return height == depth
    return generate(p_set, min_depth, max_depth, condition, ret_type)


# -------------------------------------------------------------------------------------- #
def gen_grow(p_set: PrimitiveSetTyped, min_depth: int,
             max_depth: int, ret_type=None) -> list:
    """
    Generates an expression where each leaf might have a different
    depth between *min* and *max*.

    Parameters:
        p_set: Primitive set from which primitives are selected.
        min_depth: Minimum depth of the random tree.
        max_depth: Maximum depth of the random tree.
        ret_type: The type that should return the tree when called.
            If None, the type of pset.ret is used.
    Returns:
        A grown tree with leaves at possibly different depths.
    """
    def condition(height, depth):
        cond = random.random() < p_set.terminal_ratio
        return depth == height or (depth >= min_depth and cond)
    return generate(p_set, min_depth, max_depth, condition, ret_type)


# -------------------------------------------------------------------------------------- #
def gen_half_and_half(p_set: PrimitiveSetTyped, min_depth: int,
                      max_depth: int, ret_type=None) -> list:
    """
    Generates an expression with a random choice
    between *gen_grow* and *gen_full*.

    Parameters:
        p_set: Primitive set from which primitives are selected.
        min_depth: Minimum depth of the random tree.
        max_depth: Maximum depth of the random tree.
        ret_type: The type that should return the tree when called.
            If None, the type of pset.ret is used.
    Returns:
        Either a full tree or a grown tree.
    """
    func = random.choice((gen_grow, gen_full))
    return func(p_set, min_depth, max_depth, ret_type)
