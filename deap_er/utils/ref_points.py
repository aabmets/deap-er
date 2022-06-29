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
def uniform_reference_points(n_obj, p=4, scaling=None):
    """
    Generates reference points uniformly on the hyperplane
    intersecting each axis at 1. The scaling factor is used
    to combine multiple layers of reference points.
    """
    def gen_refs_recursive(ref, n_obj_, left, total, depth):
        points = []
        if depth == n_obj_ - 1:
            ref[depth] = left / total
            points.append(ref)
        else:
            for i in range(left + 1):
                ref[depth] = i / total
                temp = gen_refs_recursive(ref.copy(), n_obj_, left - i, total, depth + 1)
                points.extend(temp)
        return points

    temp = gen_refs_recursive(numpy.zeros(n_obj), n_obj, p, p, 0)
    ref_points = numpy.array(temp)
    if scaling is not None:
        ref_points *= scaling
        ref_points += (1 - scaling) / n_obj

    return ref_points
