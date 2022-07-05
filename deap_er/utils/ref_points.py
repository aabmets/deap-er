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
import numpy


__all__ = ['uniform_reference_points']


# ====================================================================================== #
def uniform_reference_points(n_obj: int, p: int = 4,
                             scaling: float = None) -> numpy.ndarray:
    """
    Generates reference points uniformly on the hyperplane
    intersecting each axis at 1. The scaling factor is used
    to combine multiple layers of reference points.

    :param n_obj: Number of objectives.
    :param p: Number of reference points per objective.
    :param scaling: Scaling factor.
    :returns: An ndarray of reference points.
    """
    ref = numpy.zeros(n_obj)
    result = _gen_refs(ref, n_obj, p, p, 0)
    ref_points = numpy.array(result)

    if scaling is not None:
        ref_points *= scaling
        ref_points += (1 - scaling) / n_obj

    return ref_points


# -------------------------------------------------------------------------------------- #
def _gen_refs(ref, n_obj, left, total, depth) -> list:
    points = []
    if depth == n_obj - 1:
        ref[depth] = left / total
        points.append(ref)
    else:
        for i in range(left + 1):
            ref[depth] = i / total
            r_c = ref.copy()
            l_i = left - i
            d_1 = depth + 1
            result = _gen_refs(r_c, n_obj, l_i, total, d_1)
            points.extend(result)
    return points
