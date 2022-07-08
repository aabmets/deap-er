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
from .multi_list import MultiList
from .node import Node
from numpy import ndarray
import ray


__all__ = ['hypervolume', 'HyperVolume']


# ====================================================================================== #
@ray.remote
def hypervolume(point_set: ndarray, ref_point: ndarray) -> float:
    """
    Computes the hypervolume of a *point_set* around the *ref_point*
    on a local or a remote cluster using the Ray library.

    :param point_set: A set of points.
    :param ref_point: The reference point.
    :returns: The hypervolume of the point set.
    """
    hv = HyperVolume(ref_point)
    return hv.compute(point_set)


# ====================================================================================== #
class HyperVolume:
    """
    Hypervolume computation based on the third variant of the algorithm in the paper:
    "An improved dimension-sweep algorithm for the hypervolume indicator." by
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez (From the IEEE Congress on
    Evolutionary Computation, pages 1157-1163, Vancouver, Canada, July 2006).
    """
    multi_list: MultiList

    # -------------------------------------------------------------------------------------- #
    def __init__(self, ref_point: ndarray) -> None:
        """
        Creates a new HyperVolume object with *ref_point*.

        :param ref_point: The reference point for the hypervolume calculation.
        """
        self.ref_point = ref_point
        self.dims = len(ref_point)

    # -------------------------------------------------------------------------------------- #
    def compute(self, point_set: ndarray) -> float:
        """
        Computes the hypervolume that is dominated by the non-dominated *point_set*.
        Minimization is implicitly assumed.

        :param point_set: The point set for which to compute the hypervolume.
        :returns: The hypervolume of the given point set.
        """
        self._pre_process(point_set)
        return self._hv_recursive(
            self.dims - 1,
            len(point_set),
            self.dims * [-1.0e308]
        )

    # -------------------------------------------------------------------------------------- #
    def _pre_process(self, point_set: ndarray) -> None:
        if any(self.ref_point):
            point_set -= self.ref_point
        node_list = MultiList(self.dims)

        def new_node(point: tuple):
            return Node(self.dims, point)

        def sort_and_extend():
            decorated = [(node.cargo[index], node) for node in nodes]
            decorated.sort()
            nodes[:] = [node for _, node in decorated]
            node_list.extend(nodes, index)

        nodes = [new_node(p) for p in point_set]
        for index in range(self.dims):
            sort_and_extend()
        self.multi_list = node_list

    # -------------------------------------------------------------------------------------- #
    def _hv_recursive(self, dim_index: int, length: int, bounds: list) -> float:
        sentinel = self.multi_list.sentinel
        reinsert = self.multi_list.reinsert
        remove = self.multi_list.remove

        def inception():
            q.volume[dim_index] = hvol
            if q.ignore >= dim_index:
                q.area[dim_index] = q.prev[dim_index].area[dim_index]
            else:
                q.area[dim_index] = self._hv_recursive(dim_index - 1, length, bounds)
                if q.area[dim_index] <= q.prev[dim_index].area[dim_index]:
                    q.ignore = dim_index

        def in_bounds():
            a = q.prev[dim_index].cargo[dim_index] >= bounds[dim_index]
            b = q.cargo[dim_index] > bounds[dim_index]
            return True if a or b else False

        hvol = 0.0

        if length == 0:
            return hvol

        elif dim_index == 0:
            return -sentinel.next[0].cargo[0]

        elif dim_index == 1:
            q = sentinel.next[1]
            h = q.cargo[0]
            p = q.next[1]

            while p is not sentinel:
                hvol += h * (q.cargo[1] - p.cargo[1])
                if p.cargo[0] < h:
                    h = p.cargo[0]
                q = p
                p = q.next[1]

            hvol += h * q.cargo[1]
            return hvol

        else:
            p = sentinel
            q = p.prev[dim_index]

            while q.cargo is not None:
                if q.ignore < dim_index:
                    q.ignore = 0
                q = q.prev[dim_index]
            q = p.prev[dim_index]

            while length > 1 and in_bounds():
                p = q
                remove(p, dim_index, bounds)
                q = p.prev[dim_index]
                length -= 1

            if length > 1:
                hvol = q.cargo[dim_index] - q.prev[dim_index].cargo[dim_index]
                hvol *= q.prev[dim_index].area[dim_index]
                hvol += q.prev[dim_index].volume[dim_index]
            else:
                q.area[0] = 1
                q.area[1: dim_index + 1] = [
                    q.area[i] * -q.cargo[i] for i in range(dim_index)
                ]
            inception()

            while p is not sentinel:
                new_point = p.cargo[dim_index] - q.cargo[dim_index]
                hvol += q.area[dim_index] * new_point
                bounds[dim_index] = p.cargo[dim_index]
                reinsert(p, dim_index, bounds)
                length += 1
                q = p
                p = p.next[dim_index]
                inception()

            hvol -= q.area[dim_index] * q.cargo[dim_index]
            return hvol
