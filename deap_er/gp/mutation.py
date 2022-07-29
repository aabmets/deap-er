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
from typing import Callable
from inspect import isclass
import random


__all__ = [
    'mut_uniform', 'mut_node_replacement',
    'mut_ephemeral', 'mut_insert', 'mut_shrink'
]


# ====================================================================================== #
def mut_uniform(individual: GPIndividual, expr: Callable,
                prim_set: PrimitiveSetTyped) -> GPMutant:
    """
    Mutates an individual by replacing a random subtree with
    an expression generated from the given **expr**.

    :param individual: The GP tree to be mutated.
    :param expr: A callable that, when called, returns a random GP subtree.
    :param prim_set: The PrimitiveSet to be used for the mutation.
    :return: The mutated individual.

    :type individual: :ref:`GPIndividual <datatypes>`
    :rtype: :ref:`GPMutant <datatypes>`
    """
    index = random.randrange(len(individual))
    i_slice = individual.search_subtree(index)
    ret_type = individual[index].ret
    individual[i_slice] = expr(prim_set=prim_set, ret_type=ret_type)
    return individual,


# -------------------------------------------------------------------------------------- #
def mut_node_replacement(individual: GPIndividual,
                         prim_set: PrimitiveSetTyped) -> GPMutant:
    """
    Mutates an individual by replacing a random primitive
    with a random primitive from the given **p_set**.

    :param individual: The GP tree to be mutated.
    :param prim_set: The PrimitiveSet to be used for the mutation.
    :return: The mutated individual.

    :type individual: :ref:`GPIndividual <datatypes>`
    :rtype: :ref:`GPMutant <datatypes>`
    """
    if len(individual) < 2:
        return individual,

    index = random.randrange(1, len(individual))
    node = individual[index]

    if node.arity == 0:
        term = random.choice(prim_set.terminals[node.ret])
        if isclass(term):
            term = term()
        individual[index] = term
    else:
        node_ret = prim_set.primitives[node.ret]
        prims = [p for p in node_ret if p.args == node.args]
        individual[index] = random.choice(prims)

    return individual,


# -------------------------------------------------------------------------------------- #
def mut_ephemeral(individual: GPIndividual,
                  mode: str = 'all') -> GPMutant:
    """
    Mutates an individual by replacing either
    one random or all ephemeral constants.

    :param individual: The GP tree to be mutated.
    :param mode: The mode of mutation. Either 'one' or 'all'.
    :return: The mutated individual.

    :type individual: :ref:`GPIndividual <datatypes>`
    :rtype: :ref:`GPMutant <datatypes>`
    """
    if mode not in ["one", "all"]:
        raise ValueError('Mode must be one of \'one\' or \'all\'.')

    ephemera_idx = list()
    for index, node in enumerate(individual):
        if isinstance(node, Ephemeral):
            ephemera_idx.append(index)

    if len(ephemera_idx) > 0:
        if mode == "one":
            ephemera_idx = (random.choice(ephemera_idx),)

        for i in ephemera_idx:
            individual[i] = type(individual[i])()

    return individual,


# -------------------------------------------------------------------------------------- #
def mut_insert(individual: GPIndividual,
               prim_set: PrimitiveSetTyped) -> GPMutant:
    """
    Inserts a new branch at a random position in the tree.

    :param individual: The GP tree to be mutated.
    :param prim_set: The PrimitiveSet to be used for the mutation.
    :return: The mutated individual.

    :type individual: :ref:`GPIndividual <datatypes>`
    :rtype: :ref:`GPMutant <datatypes>`
    """
    index = random.randrange(len(individual))
    node = individual[index]
    slice_ = individual.search_subtree(index)
    choice = random.choice

    primitives = list()
    for p in prim_set.primitives[node.ret]:
        if node.ret in p.args:
            primitives.append(p)

    if len(primitives) == 0:
        return individual,

    new_node = choice(primitives)
    new_subtree = [None] * len(new_node.args)

    choices = list()
    for i, a in enumerate(new_node.args):
        if a == node.ret:
            choices.append(i)
    position = choice(choices)

    for i, arg_type in enumerate(new_node.args):
        if i != position:
            term = choice(prim_set.terminals[arg_type])
            if isclass(term):
                term = term()
            new_subtree[i] = term

    new_subtree[position:position + 1] = individual[slice_]
    new_subtree.insert(0, new_node)
    individual[slice_] = new_subtree

    return individual,


# -------------------------------------------------------------------------------------- #
def mut_shrink(individual: GPIndividual) -> GPMutant:
    """
    Shrinks a tree by removing a random branch, replacing
    it with a random argument of the branch.

    :param individual: The GP tree to be mutated.
    :return: The mutated individual.

    :type individual: :ref:`GPIndividual <datatypes>`
    :rtype: :ref:`GPMutant <datatypes>`
    """
    if len(individual) < 3 or individual.height <= 1:
        return individual,

    i_prims = []
    for i, node in enumerate(individual[1:], 1):
        if isinstance(node, Primitive) and node.ret in node.args:
            i_prims.append((i, node))

    if len(i_prims) != 0:
        index, prim = random.choice(i_prims)
        choices = list()
        for i, type_ in enumerate(prim.args):
            if type_ != prim.ret:
                choices.append(i)
        arg_idx = random.choice(choices)
        r_index = index + 1
        subtree = list()
        for _ in range(arg_idx + 1):
            r_slice = individual.search_subtree(r_index)
            subtree = individual[r_slice]
            r_index += len(subtree)

        i_slice = individual.search_subtree(index)
        individual[i_slice] = subtree

    return individual,
