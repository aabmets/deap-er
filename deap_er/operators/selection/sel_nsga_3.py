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
from deap_er.utilities.sorting import *
from itertools import chain
from numpy import ndarray
import numpy


__all__ = ['sel_nsga_3', 'SelNSGA3WithMemory']


# ====================================================================================== #
class SelNSGA3WithMemory:
    """
    The NSGA-III selection operator with memory for best, worst and extreme
    points. Instances of this class can be registered into a Toolbox.

    :param ref_points: Reference points for selection.
    :param sorting_algo: The algorithm to use for non-dominated
        sorting. Can be either 'log' or 'standard' string literal.
    """
    # -------------------------------------------------------- #
    def __init__(self, ref_points: ndarray, sorting_algo: str = "log"):
        self.ref_points = ref_points
        self.sorting_algo = sorting_algo
        self.best_point = numpy.full((1, ref_points.shape[1]), numpy.inf)
        self.worst_point = numpy.full((1, ref_points.shape[1]), -numpy.inf)
        self.extreme_points = None

    # -------------------------------------------------------- #
    def __call__(self, individuals: list, sel_count: int) -> list:
        """
        This method is called by the Toolbox to select
        individuals for the next generation.

        :param individuals: A list of individuals to select from.
        :param sel_count: The number of individuals to select.
        :return: A list of selected individuals.
        """
        chosen = sel_nsga_3(
            individuals, sel_count,
            self.ref_points, self.sorting_algo,
            self.best_point, self.worst_point,
            self.extreme_points, self
        )
        return chosen


# ====================================================================================== #
def sel_nsga_3(individuals: list,
               sel_count: int,
               ref_points: ndarray,
               sorting_algo: str = "log",
               best_point: ndarray = None,
               worst_point: ndarray = None,
               extreme_points: ndarray = None,
               _memory: SelNSGA3WithMemory = None) -> list:
    """
    Selects the next generation of individuals using the NSGA-III algorithm.

    :param individuals: A list of individuals to select from.
    :param sel_count: The number of individuals to select.
    :param ref_points: The reference points to use for the selection.
    :param sorting_algo: The non-dominated sorting algorithm to use, optional.
    :param best_point: Best point of the previous generation, optional.
        If not provided, finds the best point from the current individuals.
    :param worst_point: Worst point of the previous generation, optional.
        If not provided, finds the worst point from the current individuals.
    :param extreme_points: Extreme points of the previous generation, optional.
        If not provided, finds the extreme points from the current individuals.
    :param _memory: This private parameter is used by the SelNSGA3WithMemory
        objects to store the best, the worst and the extreme points of the
        selection into itself. Not recommended for manual use.
    :return: A list of selected individuals.
    """
    if sorting_algo == "standard":
        pareto_fronts = sort_non_dominated(individuals, sel_count)
    elif sorting_algo == "log":
        pareto_fronts = sort_log_non_dominated(individuals, sel_count)
    else:
        raise RuntimeError(
            f'selNSGA3: The choice of non-dominated '
            f'sorting method \'{sorting_algo}\' is invalid.'
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
    diff = sel_count - len(chosen)
    selected = _select_from_niche(
        pareto_fronts[-1], diff,
        niches[len(chosen):],
        dist[len(chosen):],
        niche_counts
    )
    chosen.extend(selected)

    if _memory and isinstance(_memory, SelNSGA3WithMemory):
        _memory.best_point = _memory.best_point.reshape((1, -1))
        _memory.worst_point = _memory.worst_point.reshape((1, -1))
        _memory.extreme_points = _memory.extreme_points
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
def _select_from_niche(individuals: list, count: int,
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
