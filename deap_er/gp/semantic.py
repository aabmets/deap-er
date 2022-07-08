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
from .generators import gen_grow
from .primitives import *
from typing import Callable
import random


__all__ = [
    'mut_semantic', 'mutSemantic',
    'cx_semantic', 'cxSemantic'
]


# ====================================================================================== #
def mut_semantic(individual: list, p_set: PrimitiveSetTyped,
                 gen_func: Callable = None, mut_step: float = None,
                 min_: int = 2, max_: int = 6) -> list:
    """
    Perform a semantic mutation on the given individual.

    :param individual: The individual to be mutated.
    :param p_set: The PrimitiveSet to be used.
    :param gen_func: The function which generates the random tree.
    :param mut_step: The mutation step.
    :param min_: Minimum depth of the random tree.
    :param max_: Maximum depth of the random tree.
    :returns: A new individual.
    """
    _check(p_set, 'mutation')

    if gen_func is None:
        gen_func = gen_grow

    if mut_step is None:
        mut_step = random.uniform(0, 2)

    tr1 = gen_func(p_set, min_, max_)
    tr2 = gen_func(p_set, min_, max_)

    tr1.insert(0, p_set.mapping['lf'])
    tr2.insert(0, p_set.mapping['lf'])

    new_ind = individual
    new_ind.insert(0, p_set.mapping["add"])
    new_ind.append(p_set.mapping["mul"])

    mutation_step = Terminal(mut_step, False, object)
    new_ind.append(mutation_step)
    new_ind.append(p_set.mapping["sub"])

    new_ind.extend(tr1)
    new_ind.extend(tr2)

    return new_ind


# -------------------------------------------------------------------------------------- #
def cx_semantic(ind1: list, ind2: list, p_set: PrimitiveSetTyped,
                gen_func: Callable = None, min_=2, max_=6) -> tuple:
    """
    Perform a semantic crossover on the given individuals.

    :param ind1: The first individual.
    :param ind2: The second individual.
    :param p_set: The PrimitiveSet to be used.
    :param gen_func: The function which generates the random tree.
    :param min_: Minimum depth of the random tree.
    :param max_: Maximum depth of the random tree.
    :returns: A tuple of two individuals.
    """
    _check(p_set, 'crossover')

    if gen_func is None:
        gen_func = gen_grow

    tr = gen_func(p_set, min_, max_)
    tr.insert(0, p_set.mapping['lf'])

    def create_ind(ind, ind_ext):
        new_ind = ind
        new_ind.insert(0, p_set.mapping["mul"])
        new_ind.insert(0, p_set.mapping["add"])
        new_ind.extend(tr)
        new_ind.append(p_set.mapping["mul"])
        new_ind.append(p_set.mapping["sub"])
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


# -------------------------------------------------------------------------------------- #
mutSemantic = deprecated('mutSemantic', mut_semantic)
cxSemantic = deprecated('cxSemantic', cx_semantic)
