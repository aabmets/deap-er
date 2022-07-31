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
from .datatypes import *
from .primitives import *
from typing import Any, Callable, Union
from functools import wraps
from copy import deepcopy
import random
import sys


__all__ = [
    'compile_tree', 'compile_adf_tree',
    'build_tree_graph', 'static_limit'
]


# ====================================================================================== #
def compile_tree(expr: GPExprTypes, prim_set: PrimitiveSetTyped) -> Any:
    """
    Evaluates the expression on the given primitive set.

    :param expr: The expression to compile. It can be a string,
        a PrimitiveTree or any object which produces a valid
        Python expression when converted into a string.
    :param prim_set: The primitive set to evaluate the expression on.
    :return: A callable if the 'p_set' has 1 or more arguments,
        otherwise the result of the evaluation.

    :type expr: :ref:`Expression <datatypes>`
    """
    code = str(expr)
    if len(prim_set.arguments) > 0:
        args = ",".join(arg for arg in prim_set.arguments)
        code = "lambda {args}: {code}".format(args=args, code=code)
    try:
        return eval(code, prim_set.context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError(
            "Recursion depth of 90 exceeded. "
            "Use bloat control on your operators.\n"
        ).with_traceback(traceback)


# -------------------------------------------------------------------------------------- #
def compile_adf_tree(expr: GPExprTypes, prim_sets: GPTypedSets) -> Any:
    """
    Compiles the expression represented by a list of trees.
    The first element of the list is the main tree, and the
    following elements are automatically defined functions
    that can be called by the first tree.

    :param expr: The expression to compile. It can be a string,
        a PrimitiveTree or any object which produces a valid
        Python expression when converted into a string.
    :param prim_sets: List of primitive sets. The first element is
        the main tree and the others are automatically defined
        functions (ADF) that can be called by the first tree.
        The last element is associated with the 'expr' and
        should contain a reference to the preceding ADFs.
    :return: A callable if the main primitive set has 1 or more
        arguments, otherwise the result of the evaluation.

    :type expr: :ref:`Expression <datatypes>`
    :type prim_sets: :ref:`PrimSets <datatypes>`
    """
    adf_dict = dict()
    func = None
    for prim_set, sub_expr in reversed(list(zip(prim_sets, expr))):
        prim_set.context.update(adf_dict)
        func = compile_tree(sub_expr, prim_set)
        adf_dict.update({prim_set.name: func})
    return func


# -------------------------------------------------------------------------------------- #
def build_tree_graph(expr: GPExprTypes) -> GPGraph:
    """
    Builds a graph representation of the given expression. The graph
    is a tuple of three elements: a list of nodes, a list of edges and a
    dictionary of node labels. The nodes are the leaves of the tree and
    the edges are the connections between the nodes. The dictionary
    contains the leaves values, where the keys are the leaves indices.

    :param expr: A tree expression to convert into a graph.
    :return: A list of nodes, a list of edges and a dictionary of labels.

    :type expr: :ref:`Expression <datatypes>`
    :rtype: :ref:`Graph <datatypes>`
    """
    nodes = list(range(len(expr)))
    edges = list()
    stack = list()
    labels = dict()

    for i, node in enumerate(expr):
        if stack:
            edges.append((stack[-1][0], i))
            stack[-1][1] -= 1
        if isinstance(node, Primitive):
            labels[i] = node.name
        else:
            labels[i] = node.value
        stack.append([i, node.arity])
        while stack and stack[-1][1] == 0:
            stack.pop()

    return nodes, edges, labels


# -------------------------------------------------------------------------------------- #
def static_limit(limiter: Callable, max_value: Union[int, float]) -> Callable:
    """
    Provides a decorator to limit the production of offspring.
    It may be used to decorate both crossover and mutation operators.
    When an invalid child is generated, it is replaced by one of its
    parents, which is randomly selected.

    :param limiter: The function which obtains the measurement from an individual.
    :param max_value: The maximum value allowed for the given measurement.
    :return: A decorator which can be applied to a GP operator in a Toolbox.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                if keep_inds and limiter(ind) > max_value:
                    new_inds[i] = random.choice(keep_inds)
            return new_inds
        return wrapper
    return decorator
