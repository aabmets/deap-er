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
from .dtypes import *
from .primitives import *
from .generators import gen_grow
from typing import Callable
import random


__all__ = ['mut_semantic', 'cx_semantic']


# ====================================================================================== #
def mut_semantic(individual: list, prim_set: PrimitiveSetTyped,
                 min_depth: int = 2, max_depth: int = 6, gen_func: Callable = None,
                 mut_step: float = None) -> tuple[list]:
    """
    Perform a semantic mutation on the given individual.

    :param individual: The individual to be mutated.
    :param prim_set: Primitive set from which primitives are selected.
    :param gen_func: The function which generates the random tree.
    :param mut_step: The mutation step.
    :param min_depth: Minimum depth of the random tree.
    :param max_depth: Maximum depth of the random tree.
    :return: The mutated individual.
    """
    _check(prim_set, 'mutation')

    if gen_func is None:
        gen_func = gen_grow

    if mut_step is None:
        mut_step = random.uniform(0, 2)

    tr1 = gen_func(prim_set, min_depth, max_depth)
    tr2 = gen_func(prim_set, min_depth, max_depth)

    tr1.insert(0, prim_set.mapping['lf'])
    tr2.insert(0, prim_set.mapping['lf'])

    new_ind = individual
    new_ind.insert(0, prim_set.mapping["add"])
    new_ind.append(prim_set.mapping["mul"])

    mutation_step = Terminal(mut_step, False, object)
    new_ind.append(mutation_step)
    new_ind.append(prim_set.mapping["sub"])

    new_ind.extend(tr1)
    new_ind.extend(tr2)

    return new_ind,


# -------------------------------------------------------------------------------------- #
def cx_semantic(ind1: list, ind2: list,
                prim_set: PrimitiveSetTyped, min_depth: int = 2,
                max_depth: int = 6, gen_func: Callable = gen_grow) -> tuple[list, list]:
    """
    Perform a semantic crossover on the given individuals.

    :param ind1: The first individual to be mated.
    :param ind2: The second individual to be mated.
    :param prim_set: Primitive set from which primitives are selected.
    :param gen_func: The function which generates the random tree.
    :param min_depth: Minimum depth of the random tree.
    :param max_depth: Maximum depth of the random tree.
    :return: Two mated individuals.
    """
    _check(prim_set, 'crossover')

    tr = gen_func(prim_set, min_depth, max_depth)
    tr.insert(0, prim_set.mapping['lf'])

    def create_ind(ind, ind_ext):
        new_ind = ind
        new_ind.insert(0, prim_set.mapping["mul"])
        new_ind.insert(0, prim_set.mapping["add"])
        new_ind.extend(tr)
        new_ind.append(prim_set.mapping["mul"])
        new_ind.append(prim_set.mapping["sub"])
        new_ind.append(Terminal(1.0, False, object))
        new_ind.extend(tr)
        new_ind.extend(ind_ext)
        return new_ind

    new_ind1 = create_ind(ind1, ind2)
    new_ind2 = create_ind(ind2, ind1)
    return new_ind1, new_ind2


# -------------------------------------------------------------------------------------- #
def _check(p_set: PrimitiveSetTyped, op: str) -> None:
    for func in ['lf', 'mul', 'add', 'sub']:
        if func not in p_set.mapping:
            raise TypeError(
                f'A \'{func}\' function is required to perform semantic \'{op}\'.'
            )
