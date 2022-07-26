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
from ray import exceptions as ray_ex
from numpy import ndarray
import numpy
import ray


__all__ = ['least_contrib']


# ====================================================================================== #
@ray.remote  # pragma: no cover
def _hvol(point_set: ndarray, ref_point: ndarray) -> float:
    hv = HyperVolume(ref_point)
    return hv.compute(point_set)


# -------------------------------------------------------------------------------------- #
def least_contrib(population: list,
                  ref_point: ndarray = None,
                  timeout: int = None) -> int:
    """
    Returns the index of the individual with the least hypervolume
    contribution. Minimization is implicitly assumed. The Ray
    multiprocessing library must be initialized by the user with
    :code:`ray.init()` before calls to this function can be made.

    :param population: A list of non-dominated individuals,
        where each individual has a Fitness attribute.
    :param ref_point: The reference point for the hypervolume, optional.
    :param timeout: Timeout for the computation. Defaults to 60 seconds.
    :raise TimeoutError: If the computation times out.
    :return: The index of the individual with the least hypervolume contribution.
    """
    if not ray.is_initialized():
        raise RuntimeError(
            'The user must initialize the Ray library with ray.init() '
            'before any calls to the least_contrib function can be made.'
        )

    wvals = [ind.fitness.wvalues for ind in population]
    wvals = numpy.array(wvals) * -1
    if ref_point is None:
        ref_point = numpy.max(wvals, axis=0) + 1

    object_refs = []
    for i in range(len(population)):
        front = (wvals[:i], wvals[i + 1:])
        front = numpy.concatenate(front)
        object_ref = _hvol.remote(front, ref_point)
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
