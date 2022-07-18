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
from deap_er.datatypes import Individual
from .hypervolume import HyperVolume

from ray import exceptions as ray_ex
import ray

from numpy import ndarray
import numpy


__all__ = ['least_contrib']


# ====================================================================================== #
@ray.remote  # pragma: no cover
def _hvol(point_set: Individual, ref_point: Individual) -> float:
    hv = HyperVolume(ref_point)
    return hv.compute(point_set)


# -------------------------------------------------------------------------------------- #
def least_contrib(population: Individual,
                  ref: ndarray = None,
                  timeout: int = None) -> int:
    """
    Returns the index of the individual with the least hypervolume
    contribution. The hypervolume is computed on a local or a remote
    cluster using the Ray multiprocessing library, which must be
    manually initialized by the user before this function can be used.

    Parameters:
        population: A sequence of non-dominated individuals,
            where each individual has a Fitness attribute.
        ref: The reference point for the hypervolume, optional.
        timeout: The timeout for the computation. Defaults to
            60 seconds. Raises a TimeoutError if the computation
            does not finish within the given timeout.
    Returns:
        The index of the individual with the least hypervolume contribution.
    """
    if not ray.is_initialized():
        raise RuntimeError(
            'The user must initialize the Ray library with ray.init() '
            'before any calls to the least_contrib function can be made.'
        )

    wvals = [ind.fitness.wvalues for ind in population]
    wvals = numpy.array(wvals) * -1
    if ref is None:
        ref = numpy.max(wvals, axis=0) + 1

    object_refs = []
    for i in range(len(population)):
        front = (wvals[:i], wvals[i + 1:])
        front = numpy.concatenate(front)
        object_ref = _hvol.remote(front, ref)
        object_refs.append(object_ref)

    args = dict(object_refs=object_refs)
    if timeout:
        args['timeout'] = float(timeout)
    try:
        contrib_values: list = ray.get(**args)
    except ray_ex.GetTimeoutError as e:
        ray.cancel(object_refs, force=True)
        raise TimeoutError(
            f'Indicator \'least_contrib\' hypervolume calculation '
            f'exceeded the timeout of {timeout} seconds.'
        ).with_traceback(e.__traceback__)

    argmax = numpy.argmax(contrib_values)
    return int(argmax)
