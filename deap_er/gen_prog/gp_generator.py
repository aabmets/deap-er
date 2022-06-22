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
from .gp_primitives import PrimitiveSetTyped
from typing import Callable
from inspect import isclass
import random
import sys


__all__ = [
    'gen_full', 'genFull',
    'gen_grow', 'genGrow',
    'gen_half_and_half', 'genHalfAndHalf', 'genRamped',
    'generate'
]


# ====================================================================================== #
def gen_full(p_set: PrimitiveSetTyped,
             min_: int, max_: int, type_=None) -> list:

    def condition(height, depth):
        return height == depth
    return generate(p_set, min_, max_, condition, type_)


# -------------------------------------------------------------------------------------- #
def gen_grow(p_set: PrimitiveSetTyped,
             min_: int, max_: int, type_=None) -> list:

    def condition(height, depth):
        cond = random.random() < p_set.terminalRatio
        return depth == height or (depth >= min_ and cond)
    return generate(p_set, min_, max_, condition, type_)


# -------------------------------------------------------------------------------------- #
def gen_half_and_half(p_set: PrimitiveSetTyped,
                      min_: int, max_: int, type_=None) -> list:

    func = random.choice((genGrow, genFull))
    return func(p_set, min_, max_, type_)


# -------------------------------------------------------------------------------------- #
def generate(p_set: PrimitiveSetTyped, min_: int,
             max_: int, condition: Callable, type_=None) -> list:

    err_msg = f'The gp.generate function tried to add a terminal ' \
              f'of type \'{type_}\', but there is none available.'
    if type_ is None:
        type_ = p_set.ret
    expr = list()
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                term = random.choice(p_set.terminals[type_])
                if isclass(term):
                    term = term()
                expr.append(term)
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(err_msg).with_traceback(traceback)
        else:
            try:
                prim = random.choice(p_set.primitives[type_])
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(err_msg).with_traceback(traceback)
    return expr


# -------------------------------------------------------------------------------------- #
genFull = deprecated('genFull', gen_full)
genGrow = deprecated('genGrow', gen_grow)
genHalfAndHalf = deprecated('genHalfAndHalf', gen_half_and_half)
genRamped = deprecated('genRamped', gen_half_and_half)
