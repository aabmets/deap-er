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
from .sel_various import sel_random
from operator import attrgetter
from functools import partial
import random


__all__ = ['sel_tournament_dcd', 'selTournamentDCD']


# ====================================================================================== #
def sel_tournament(individuals, k, tournsize, fit_attr="fitness"):
    chosen = []
    for i in range(k):
        aspirants = sel_random(individuals, tournsize)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen


# -------------------------------------------------------------------------------------- #
def sel_double_tournament(individuals, k, fitness_size, parsimony_size, fitness_first, fit_attr="fitness"):
    assert (1 <= parsimony_size <= 2), "Parsimony tournament size has to be in the range [1, 2]."

    def _size_tourney(select):
        chosen = []
        for i in range(k):
            prob = parsimony_size / 2.
            ind1, ind2 = select(individuals, k=2)

            if len(ind1) > len(ind2):
                ind1, ind2 = ind2, ind1
            elif len(ind1) == len(ind2):
                prob = 0.5

            chosen.append(ind1 if random.random() < prob else ind2)

        return chosen

    def _fit_tourney(select):
        chosen = []
        for i in range(k):
            aspirants = select(individuals, k=fitness_size)
            chosen.append(max(aspirants, key=attrgetter(fit_attr)))
        return chosen

    if fitness_first:
        t_fit = partial(_fit_tourney, select=sel_random)
        return _size_tourney(t_fit)
    else:
        t_size = partial(_size_tourney, select=sel_random)
        return _fit_tourney(t_size)


# -------------------------------------------------------------------------------------- #
def sel_tournament_dcd(individuals, k):
    if k > len(individuals):
        err_msg = "sel_tournament_dcd: k must be less than or equal to individuals length."
        raise ValueError(err_msg)

    if k == len(individuals) and k % 4 != 0:
        err_msg = 'sel_tournament_dcd: k must be divisible by four if k == len(individuals)'
        raise ValueError(err_msg)

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

    individuals_1 = random.sample(individuals, len(individuals))
    individuals_2 = random.sample(individuals, len(individuals))

    chosen = []
    for i in range(0, k, 4):
        chosen.append(tourney(individuals_1[i],   individuals_1[i+1]))
        chosen.append(tourney(individuals_1[i+2], individuals_1[i+3]))
        chosen.append(tourney(individuals_2[i],   individuals_2[i+1]))
        chosen.append(tourney(individuals_2[i+2], individuals_2[i+3]))

    return chosen


# -------------------------------------------------------------------------------------- #
selTournament = deprecated('selTournament', sel_tournament)
selDoubleTournament = deprecated('selDoubleTournament', sel_double_tournament)
selTournamentDCD = deprecated('selTournamentDCD', sel_tournament_dcd)
