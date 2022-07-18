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
from deap_er.datatypes import Subscript
from .sel_various import sel_random
from operator import attrgetter
from functools import partial
import random


__all__ = ['sel_tournament', 'sel_double_tournament', 'sel_tournament_dcd']


# ====================================================================================== #
def sel_tournament(individuals: Subscript, rounds: int,
                   contestants: int, fit_attr: str = "fitness") -> list:
    """
    Selects the best individual among the randomly
    chosen *contestants* for *rounds* times.

    :param individuals: A list of individuals to select from.
    :param rounds: The number of rounds in the tournament.
    :param contestants: The number of individuals
        participating in each fitness tournament.
    :param fit_attr: The attribute of individuals
        to use as selection criterion.
    :returns: A list of selected individuals.
    """
    chosen = []
    for i in range(rounds):
        aspirants = sel_random(individuals, contestants)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen


# -------------------------------------------------------------------------------------- #
def sel_double_tournament(individuals: Subscript, rounds: int,
                          fitness_size: int, parsimony_size: int,
                          fitness_first: bool, fit_attr: str = "fitness") -> list:
    """
    Tournament selection which uses the size of the individuals in
    order to discriminate good solutions. It can also be used for
    Genetic Programming as a bloat control technique.

    :param individuals: A list of individuals to select from.
    :param rounds: The number of rounds in the tournament.
    :param fitness_size: The number of individuals
        participating in each fitness tournament.
    :param parsimony_size: The number of individuals participating
        in each size tournament. This value has to be
        a real number in the range of [1,2].
    :param fitness_first: If set to True, the fitness
        tournament will be performed first.
    :param fit_attr: The attribute of individuals
        to use as selection criterion.
    :returns: A list of selected individuals.
    """
    def _size_tourney(select):
        chosen = []
        for i in range(rounds):
            prob = parsimony_size / 2.
            ind1, ind2 = select(individuals, count=2)
            if len(ind1) > len(ind2):
                ind1, ind2 = ind2, ind1
            elif len(ind1) == len(ind2):
                prob = 0.5
            chosen.append(ind1 if random.random() < prob else ind2)
        return chosen

    def _fit_tourney(select):
        chosen = []
        for i in range(rounds):
            aspirants = select(individuals, count=fitness_size)
            chosen.append(max(aspirants, key=attrgetter(fit_attr)))
        return chosen

    if not (1 <= parsimony_size <= 2):
        raise ValueError("Parsimony tournament size has to be in the range of [1, 2].")

    if fitness_first:
        t_fit = partial(_fit_tourney, select=sel_random)
        return _size_tourney(t_fit)
    else:
        t_size = partial(_size_tourney, select=sel_random)
        return _fit_tourney(t_size)


# -------------------------------------------------------------------------------------- #
def sel_tournament_dcd(individuals: Subscript, count: int) -> list:
    """
    Tournament selection based on the dominance between two individuals,
    if the two individuals do not inter-dominate, then the selection
    is made based on their crowding distance. The *individuals* sequence
    length has to be a multiple of four only if the *count* is equal to
    the length of *individuals*. This selection requires the individuals
    to have the *crowding_dist* attribute, which can be set by the
    *assign_crowding_dist* function.

    :param individuals: A list of individuals to select from.
    :param count: The number of individuals to select.
    :returns: A list of selected individuals.
    """
    def tourney(ind1, ind2):
        if ind1.fitness.dominates(ind2.fitness):
            return ind1
        elif ind2.fitness.dominates(ind1.fitness):
            return ind2
        if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
            return ind2
        elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
            return ind1
        if random.random() <= 0.5:
            return ind1
        return ind2

    if count > len(individuals):
        raise ValueError("count must be less than or equal to individuals length.")

    if count == len(individuals) and count % 4 != 0:
        raise ValueError("count must be divisible by four if k == len(individuals)")

    individuals_1 = random.sample(individuals, len(individuals))
    individuals_2 = random.sample(individuals, len(individuals))

    chosen = []
    for i in range(0, count, 4):
        chosen.append(tourney(individuals_1[i],   individuals_1[i+1]))
        chosen.append(tourney(individuals_1[i+2], individuals_1[i+3]))
        chosen.append(tourney(individuals_2[i],   individuals_2[i+1]))
        chosen.append(tourney(individuals_2[i+2], individuals_2[i+3]))

    return chosen
