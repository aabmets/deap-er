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
from collections.abc import Sequence, Callable
from deap_er._datatypes import NumOrSeq
from itertools import repeat
from functools import wraps


__all__ = ['DeltaPenalty', 'ClosestValidPenalty']


# ====================================================================================== #
class DeltaPenalty:
    """
    This decorator returns penalized fitness for invalid individuals and
    the original fitness value for valid individuals. The penalized fitness
    is made of a constant factor *delta* added with an optional *distance*
    penalty. The distance function, if provided, returns a value, which is
    growing as the individual moves away from the valid zone.

    :param feasibility: A function returning the
        validity status of any individual.
    :param delta: Constant or a sequence of constants
        returned for an invalid individual.
    :param distance: A function returning the distance
        between the individual and a given valid point.
    :returns: A decorator for the fitness function.
    """
    def __init__(self, feasibility: Callable,
                 delta: NumOrSeq, distance: Callable = None):
        self.fea_func = feasibility
        if not isinstance(delta, Sequence):
            self.delta = repeat(delta)
        else:
            self.delta = delta
        self.dist_fct = distance

    # -------------------------------------------------------------------------------------- #
    def __call__(self, func):
        @wraps(func)
        def wrapper(individual, *args, **kwargs):
            if self.fea_func(individual):
                return func(individual, *args, **kwargs)

            weights = tuple(1 if w >= 0 else -1 for w in individual.fitness.weights)

            dists = [0 for _ in individual.fitness.weights]
            if self.dist_fct is not None:
                dists = self.dist_fct(individual)
                if not isinstance(dists, Sequence):
                    dists = repeat(dists)

            return tuple(d - w * dist for d, w, dist in zip(self.delta, weights, dists))

        return wrapper


# ====================================================================================== #
class ClosestValidPenalty:
    """
    This decorator returns penalized fitness for invalid individuals and
    the original fitness value for valid individuals. The penalized fitness
    is made of the fitness of the closest valid individual added with an
    optional weighted *distance* penalty. The distance function, if
    provided, returns a value, which is growing as the individual
    moves away from the valid zone.

    :param feasibility: A function returning the
        validity status of any individual.
    :param feasible: A function returning the closest feasible
        individual from the current invalid individual.
    :param alpha: Multiplication factor on the distance
        between the valid and invalid individual.
    :param distance: A function returning the distance
        between the individual and a given valid point.
    :returns: A decorator for the fitness function.
    """
    # -------------------------------------------------------------------------------------- #
    def __init__(self, feasibility, feasible, alpha, distance=None):
        self.fea_func = feasibility
        self.fbl_fct = feasible
        self.alpha = alpha
        self.dist_fct = distance

    # -------------------------------------------------------------------------------------- #
    def __call__(self, func):
        @wraps(func)
        def wrapper(individual, *args, **kwargs):
            if self.fea_func(individual):
                return func(individual, *args, **kwargs)

            f_ind = self.fbl_fct(individual)
            f_fbl = func(f_ind, *args, **kwargs)

            weights = tuple(1.0 if w >= 0 else -1.0 for w in individual.fitness.weights)

            if len(weights) != len(f_fbl):
                raise IndexError("Fitness weights and computed fitness are of different size.")

            dists = [0 for _ in individual.fitness.weights]
            if self.dist_fct is not None:
                dists = self.dist_fct(f_ind, individual)
                if not isinstance(dists, Sequence):
                    dists = repeat(dists)

            return tuple(f - w * self.alpha * d for f, w, d in zip(f_fbl, weights, dists))

        return wrapper
