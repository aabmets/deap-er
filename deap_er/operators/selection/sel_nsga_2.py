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
from deap_er.utilities.sorting import *
from .sel_helpers import assign_crowding_dist
from operator import attrgetter
from itertools import chain


__all__ = ['sel_nsga_2']


# ====================================================================================== #
def sel_nsga_2(individuals: Subscript, count: int,
               nd_algo: str = 'standard') -> list:
    """
    Selects the next generation of individuals using the NSGA-II algorithm.
    Usually, the size of *individuals* should be larger than the *count*
    parameter. If the size of *individuals* is equal to *count*, the
    population will be sorted according to their pareto fronts.

    :param individuals: A list of individuals to select from.
    :param count: The number of individuals to select.
    :param nd_algo: The algorithm to use for non-dominated sorting.
    :returns: A list of selected individuals.
    """
    if nd_algo == 'standard':
        pareto_fronts = sort_non_dominated(individuals, count)
    elif nd_algo == 'log':
        pareto_fronts = sort_log_non_dominated(individuals, count)
    else:
        raise RuntimeError(
            f'selNSGA2: The choice of non-dominated '
            f'sorting method \'{nd_algo}\' is invalid.'
        )

    for front in pareto_fronts:
        assign_crowding_dist(front)

    chosen = list(chain(*pareto_fronts[:-1]))
    count = count - len(chosen)
    if count > 0:
        attr = attrgetter("fitness.crowding_dist")
        sorted_front = sorted(pareto_fronts[-1], key=attr, reverse=True)
        chosen.extend(sorted_front[:count])

    return chosen
