#
#   MIT License
#   
#   Copyright (c) 2022, Mattias Aabmets
#   
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#   
#   SPDX-License-Identifier: MIT
#
from .hypervolume import HyperVolume
from typing import Callable, Optional, Union
import numpy


__all__ = ['least_contrib']


# ====================================================================================== #
def _compute_hv(data: tuple) -> float:
    point_set, ref_point = data[0], data[1]
    hv = HyperVolume(ref_point)
    return hv.compute(point_set)


# -------------------------------------------------------------------------------------- #
def least_contrib(population: list, ref_point: Optional[list] = None,
                  map_func: Optional[Callable] = map) -> Union[int, numpy.ndarray]:
    """
    Returns the index of the individual with the least hypervolume
    contribution. Minimization is implicitly assumed.

    :param population: A list of non-dominated individuals,
        where each individual has a Fitness attribute.
    :param ref_point: The reference point for the hypervolume, optional.
    :param map_func: Any map function which maps an iterable to a callable,
        optional. This can be used to speed up the computation by providing
        a multiprocess mapping function which is associated to a pool of
        workers. The default is the regular single-process map function.
    :return: The index of the individual with the least hypervolume contribution.
    """
    wvals = [ind.fitness.wvalues for ind in population]
    wvals = numpy.array(wvals) * -1
    if ref_point is None:
        ref_point = numpy.max(wvals, axis=0) + 1
    else:
        ref_point = numpy.array(ref_point)

    data = []
    for i in range(len(population)):
        point_set = (wvals[:i], wvals[i + 1:])
        point_set = numpy.concatenate(point_set)
        data.append((point_set, ref_point))

    contrib_values = list(map_func(_compute_hv, data))
    return numpy.argmax(contrib_values)
