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
from .hypervolume import HyperVolume
import numpy


__all__ = ['least_contrib']


# ====================================================================================== #
def least_contrib(population: list, ref_point: list = None) -> int:
    """
    Returns the index of the individual with the least hypervolume
    contribution. Minimization is implicitly assumed. The Ray
    multiprocessing library must be initialized by the user with
    :code:`ray.init()` before calls to this function can be made.

    :param population: A list of non-dominated individuals,
        where each individual has a Fitness attribute.
    :param ref_point: The reference point for the hypervolume, optional.
    :raise TimeoutError: If the computation times out.
    :return: The index of the individual with the least hypervolume contribution.
    """
    wvals = [ind.fitness.wvalues for ind in population]
    wvals = numpy.array(wvals) * -1
    if ref_point is None:
        ref_point = numpy.max(wvals, axis=0) + 1
    else:
        ref_point = numpy.array(ref_point)

    contrib_values = []
    for i in range(len(population)):
        point_set = (wvals[:i], wvals[i + 1:])
        point_set = numpy.concatenate(point_set)
        hv = HyperVolume(ref_point)
        contrib = hv.compute(point_set)
        contrib_values.append(contrib)

    argmax = numpy.argmax(contrib_values)
    return int(argmax)
