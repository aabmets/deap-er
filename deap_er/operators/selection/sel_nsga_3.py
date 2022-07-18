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
from itertools import chain
from numpy import ndarray
import numpy


__all__ = ['sel_nsga_3', 'SelNSGA3WithMemory']


# ====================================================================================== #
class SelNSGA3WithMemory:

    def __init__(self, ref_points: ndarray, nd_algo: str = "log") -> None:
        """
        The NSGA-III selection operator with memory for best,
        worst and extreme points. Instances of this class
        can be registered into a Toolbox.

        :param ref_points: Reference points for selection.
        :param nd_algo: Name of the non-dominated sorting algorithm.
        """
        self.ref_points = ref_points
        self.nd_algo = nd_algo
        self.best_point = numpy.full((1, ref_points.shape[1]), numpy.inf)
        self.worst_point = numpy.full((1, ref_points.shape[1]), -numpy.inf)
        self.extreme_points = None

    # -------------------------------------------------------- #
    def __call__(self, individuals: Subscript, count: int) -> list:
        """
        This method is called by the Toolbox to select
        individuals for the next generation.

        :param individuals: A list of individuals to select from.
        :param count: The number of individuals to select.
        :returns: A list of selected individuals.
        """
        chosen = sel_nsga_3(
            individuals, count, self.ref_points, self.nd_algo,
            self.best_point, self.worst_point,
            self.extreme_points, self
        )
        return chosen


# ====================================================================================== #
def sel_nsga_3(individuals: Subscript, count: int,
               ref_points: ndarray, nd_algo: str = "log",
               best_point: ndarray = None,
               worst_point: ndarray = None,
               extreme_points: ndarray = None,
               memory: SelNSGA3WithMemory = None) -> list:
    """
    Selects the next generation of individuals using the NSGA-III algorithm.

    :param individuals: A list of individuals to select from.
    :param count: The number of individuals to select.
    :param ref_points: The reference points to use for the selection.
    :param nd_algo: The non-dominated sorting algorithm to use.
    :param best_point: Best point of the previous generation. If not
        provided, finds the best point from the current individuals.
    :param worst_point: Worst point of the previous generation. If not
        provided, finds the worst point from the current individuals.
    :param extreme_points: Extreme points of the previous generation. If not
        provided, finds the extreme points from the current individuals.
    :param memory: It is used by a SelNSGA3WithMemory instance to store the
        best, the worst and the extreme points of the selection into itself.
        Manual usage of this parameter is not recommended.
    :returns: A tuple of selected individuals and a memory object.
    """
    if nd_algo == "standard":
        pareto_fronts = sort_non_dominated(individuals, count)
    elif nd_algo == "log":
        pareto_fronts = sort_log_non_dominated(individuals, count)
    else:
        raise RuntimeError(
            f'selNSGA3: The choice of non-dominated '
            f'sorting method \'{nd_algo}\' is invalid.'
        )

    fitness = numpy.array([ind.fitness.wvalues for f in pareto_fronts for ind in f])
    fitness *= -1

    if best_point is not None and worst_point is not None:
        best_point = numpy.min(numpy.concatenate((fitness, best_point), axis=0), axis=0)
        worst_point = numpy.max(numpy.concatenate((fitness, worst_point), axis=0), axis=0)
    else:
        best_point = numpy.min(fitness, axis=0)
        worst_point = numpy.max(fitness, axis=0)

    extreme_points = _find_extreme_points(fitness, best_point, extreme_points)
    front_worst = numpy.max(fitness[:sum(len(f) for f in pareto_fronts), :], axis=0)
    intercepts = _find_intercepts(extreme_points, best_point, worst_point, front_worst)
    niches, dist = _associate_to_niche(fitness, ref_points, best_point, intercepts)

    niche_counts = numpy.zeros(len(ref_points), dtype=numpy.int64)
    index, counts = numpy.unique(niches[:-len(pareto_fronts[-1])], return_counts=True)
    niche_counts[index] = counts

    chosen = list(chain(*pareto_fronts[:-1]))
    sel_count = len(chosen)
    diff = count - sel_count
    selected = _select_from_niche(
        pareto_fronts[-1], diff,
        niches[sel_count:],
        dist[sel_count:],
        niche_counts
    )
    chosen.extend(selected)

    if memory and isinstance(memory, SelNSGA3WithMemory):
        memory.best_point = memory.best_point.reshape((1, -1))
        memory.worst_point = memory.worst_point.reshape((1, -1))
        memory.extreme_points = memory.extreme_points
    return chosen


# -------------------------------------------------------------------------------------- #
def _find_extreme_points(fitness: ndarray, best_point: ndarray,
                         extreme_points: ndarray = None) -> ndarray:

    if extreme_points is not None:
        fitness = numpy.concatenate((fitness, extreme_points), axis=0)

    ft = fitness - best_point
    asf = numpy.eye(best_point.shape[0])
    asf[asf == 0] = 1e6
    asf = numpy.max(ft * asf[:, numpy.newaxis, :], axis=2)

    min_asf_idx = numpy.argmin(asf, axis=1)
    return fitness[min_asf_idx, :]


# -------------------------------------------------------------------------------------- #
def _find_intercepts(extreme_points: ndarray, best_point: ndarray,
                     current_worst: ndarray, front_worst: ndarray) -> ndarray:

    b = numpy.ones(extreme_points.shape[1])
    big_a = extreme_points - best_point
    try:
        x = numpy.linalg.solve(big_a, b)
    except numpy.linalg.LinAlgError:
        intercepts = current_worst
    else:
        if numpy.count_nonzero(x) != len(x):
            intercepts = front_worst
        else:
            intercepts = 1 / x

            if (not numpy.allclose(numpy.dot(big_a, x), b) or
                    numpy.any(intercepts <= 1e-6) or
                    numpy.any((intercepts + best_point) > current_worst)):
                intercepts = front_worst

    return intercepts


# -------------------------------------------------------------------------------------- #
def _associate_to_niche(fitness: ndarray, reference_points: ndarray,
                        best_point: ndarray, intercepts: ndarray) -> tuple:
    fn = (fitness - best_point) / (intercepts - best_point)
    fn = numpy.repeat(numpy.expand_dims(fn, axis=1), len(reference_points), axis=1)
    norm = numpy.linalg.norm(reference_points, axis=1)

    distances = numpy.sum(fn * reference_points, axis=2) / norm.reshape(1, -1)
    dist_1 = distances[:, :, numpy.newaxis]
    dist_2 = reference_points[numpy.newaxis, :, :]
    dist_3 = norm[numpy.newaxis, :, numpy.newaxis]
    distances = dist_1 * dist_2 / dist_3
    distances = numpy.linalg.norm(distances - fn, axis=2)

    niches = numpy.argmin(distances, axis=1)
    distances = distances[list(range(niches.shape[0])), niches]
    return niches, distances


# -------------------------------------------------------------------------------------- #
def _select_from_niche(individuals: Subscript, count: int,
                       niches: ndarray, distances: ndarray,
                       niche_counts: ndarray) -> list:
    selected = []
    available = numpy.ones(len(individuals), dtype=numpy.bool)
    while len(selected) < count:
        n = count - len(selected)

        available_niches = numpy.zeros(len(niche_counts), dtype=numpy.bool)
        available_niches[numpy.unique(niches[available])] = True
        min_count = numpy.min(niche_counts[available_niches])

        logical_and = numpy.logical_and(available_niches, niche_counts == min_count)
        selected_niches = numpy.flatnonzero(logical_and)
        numpy.random.shuffle(selected_niches)
        selected_niches = selected_niches[:n]

        for niche in selected_niches:
            logical_and = numpy.logical_and(niches == niche, available)
            niche_individuals = numpy.flatnonzero(logical_and)
            numpy.random.shuffle(niche_individuals)

            if niche_counts[niche] == 0:
                arg_min = numpy.argmin(distances[niche_individuals])
                sel_index = niche_individuals[arg_min]
            else:
                sel_index = niche_individuals[0]

            available[sel_index] = False
            niche_counts[niche] += 1
            selected.append(individuals[sel_index])

    return selected
