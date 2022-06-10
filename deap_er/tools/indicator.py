# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 The Original DEAP Team, Mattias Aabmets and Contributors          #
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
from hyper_volume import hypervolume as compute_hv
import numpy


__all__ = ["hypervolume"]


# ====================================================================================== #
def hypervolume(front, **kwargs):
    """
    Returns the index of the individual with the least the hypervolume
    contribution. The provided *front* should be a set of non-dominated
    individuals having each a :attr:`fitness` attribute.
    """
    # Must use w_values * -1 since hypervolume uses implicit
    # minimization and minimization in deap use max on -obj
    w_obj = numpy.array([ind.fitness.wvalues for ind in front]) * -1
    ref_point = kwargs.get("ref", None)
    if ref_point is None:
        ref_point = numpy.max(w_obj, axis=0) + 1

    def contribution(i):
        # The contribution of point p_i in point set P
        # is the hypervolume of P without p_i
        point_set = numpy.concatenate((w_obj[:i], w_obj[i+1:]))
        return compute_hv(point_set, ref_point)

    # Parallelization note: Cannot pickle local function
    contrib_values = list(map(contribution, list(range(len(front)))))

    # Select the maximum hypervolume value (correspond to the minimum difference)
    return numpy.argmax(contrib_values)
