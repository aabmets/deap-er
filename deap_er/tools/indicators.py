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
from .hypervolume import hypervolume
from deap_er._datatypes import SetItemSeq
from ray.exceptions import GetTimeoutError
from numpy import ndarray
import numpy
import ray


__all__ = ['least_contrib']


# ====================================================================================== #
def least_contrib(population: SetItemSeq,
                  ref: ndarray = None,
                  timeout: int = None) -> int:
    """
    Returns the index of the individual with the least
    hypervolume contribution. The *population* argument
    should be a sequence of non-dominated individuals,
    where each has a *fitness* attribute.

    :param population: A sequence of individuals.
    :param ref: The reference point for the hypervolume, optional.
    :param timeout: The timeout for the computation.
        Defaults to 60 seconds. Raises a TimeoutError if the
        computation does not finish within the given timeout.
    :returns: The index of the individual with
        the least hypervolume contribution.
    """
    if not ray.is_initialized():
        raise RuntimeError(
            'Ray must be initialized with ray.init() before any '
            'calls to the least_contrib indicator can be made.'
        )
    wvals = [ind.fitness.wvalues for ind in population]
    wvals = numpy.array(wvals) * -1
    if ref is None:
        ref = numpy.max(wvals, axis=0) + 1

    object_refs = []
    for i in range(len(population)):
        front = (wvals[:i], wvals[i + 1:])
        front = numpy.concatenate(front)
        object_ref = hypervolume.remote(front, ref)
        object_refs.append(object_ref)

    args = dict(object_refs=object_refs)
    if timeout:
        args['timeout'] = float(timeout)
    try:
        contrib_values: list = ray.get(**args)
    except GetTimeoutError as e:
        ray.cancel(object_refs, force=True)
        raise TimeoutError(
            f'Indicator \'least_contrib\' hypervolume calculation '
            f'exceeded the timeout of {timeout} seconds.'
        ).with_traceback(e.__traceback__)

    argmax = numpy.argmax(contrib_values)
    return int(argmax)
