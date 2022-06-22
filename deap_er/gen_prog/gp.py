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
from deap_er._deprecated import deprecated
from .gp_primitives import *
from typing import Any, Callable
from functools import wraps
import random
import copy
import sys


Expr = PrimitiveTree | str
PSets = list[PrimitiveSetTyped]
Graph = tuple[list, list, dict]


__all__ = [
    'compile_tree', 'compile',
    'compile_adf_tree', 'compileADF',
    'build_tree_graph', 'graph',
    'static_limit', 'staticLimit'
]


# ====================================================================================== #
def compile_tree(expr: Expr, p_set: PrimitiveSetTyped) -> Any:
    code = str(expr)
    if len(p_set.arguments) > 0:
        args = ",".join(arg for arg in p_set.arguments)
        code = "lambda {args}: {code}".format(args=args, code=code)
    try:
        return eval(code, p_set.context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError(
            "DEAP : Error in tree evaluation : "
            "Python cannot evaluate a tree higher than 90. "
            "To avoid this problem, you should use bloat control on your operators. "
            "See the DEAP documentation for more information. "
            "DEAP will now abort."
        ).with_traceback(traceback)


# -------------------------------------------------------------------------------------- #
def compile_adf_tree(expr: Expr, p_sets: PSets) -> Callable:
    adf_dict = dict()
    func = None
    for p_set, sub_expr in reversed(list(zip(p_sets, expr))):
        p_set.context.update(adf_dict)
        func = compile_tree(sub_expr, p_set)
        adf_dict.update({p_set.name: func})
    return func


# -------------------------------------------------------------------------------------- #
def build_tree_graph(expr: Expr) -> Graph:
    nodes = list(range(len(expr)))
    edges = list()
    stack = list()
    labels = dict()

    for i, node in enumerate(expr):
        if stack:
            edges.append((stack[-1][0], i))
            stack[-1][1] -= 1
        cond = isinstance(node, Primitive)
        labels[i] = node.name if cond else node.value
        stack.append([i, node.arity])
        while stack and stack[-1][1] == 0:
            stack.pop()

    return nodes, edges, labels


# -------------------------------------------------------------------------------------- #
def static_limit(key: Callable, max_value: int | float) -> Callable:
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_individuals = [copy.deepcopy(ind) for ind in args]
            new_individuals = list(func(*args, **kwargs))
            for i, ind in enumerate(new_individuals):
                if key(ind) > max_value:
                    choice = random.choice(keep_individuals)
                    new_individuals[i] = choice
            return new_individuals
        return wrapper
    return decorator


# ====================================================================================== #
compile = deprecated('compile', compile_tree)
compileADF = deprecated('compileADF', compile_adf_tree)
graph = deprecated('graph', build_tree_graph)
staticLimit = deprecated('staticLimit', static_limit)
