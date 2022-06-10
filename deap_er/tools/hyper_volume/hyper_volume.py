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
from multi_list import MultiList
from node import Node


# ====================================================================================== #
class HyperVolume:
    """
    Hypervolume computation based on variant 3 of the algorithm in the paper:
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved dimension-sweep
    algorithm for the hyper_volume indicator. In IEEE Congress on Evolutionary
    Computation, pages 1157-1163, Vancouver, Canada, July 2006.
    Minimization is implicitly assumed here!
    """
    multi_list: MultiList

    # -------------------------------------------------------------------------------------- #
    def __init__(self, ref_point):
        self.ref_point = ref_point

    # -------------------------------------------------------------------------------------- #
    def sortByDimension(self, nodes, i):
        """
        Sorts the list of nodes by the i-th value of the contained points.
        """
        decorated = [(node.cargo[i], node) for node in nodes]
        decorated.sort()
        nodes[:] = [node for (_, node) in decorated]

    # -------------------------------------------------------------------------------------- #
    def preProcess(self, front):
        """
        Sets up the list data structure needed for calculation.
        """
        dimensions = len(self.ref_point)
        node_list = MultiList(dimensions)
        nodes = [Node(dimensions, point) for point in front]
        for i in range(dimensions):
            self.sortByDimension(nodes, i)
            node_list.extend(nodes, i)
        self.multi_list = node_list

    # -------------------------------------------------------------------------------------- #
    def compute(self, point_set):
        """
        Returns the hypervolume that is dominated by a non-dominated front.
        Before the HV computation, front and reference point are translated, so
        that the reference point is [0, ..., 0].
        """
        if any(self.ref_point):
            point_set -= self.ref_point
        self.preProcess(point_set)

        dimensions = len(self.ref_point)
        bounds = [-1.0e308] * dimensions
        hyper_volume = self.hvRecursive(
            dimensions - 1, len(point_set),
            bounds
        )
        return hyper_volume

    # -------------------------------------------------------------------------------------- #
    def hvRecursive(self, dim_index, length, bounds):
        """
        Recursive call to hypervolume calculation.
        In contrast to the paper, the code assumes that the reference point
        is [0, ..., 0]. This allows the avoidance of a few operations.
        """
        hv_recursive = self.hvRecursive
        sentinel = self.multi_list.sentinel
        reinsert = self.multi_list.reinsert
        remove = self.multi_list.remove

        h_vol = 0.0

        if length == 0:
            return h_vol

        elif dim_index == 0:
            return -sentinel.next[0].cargo[0]

        elif dim_index == 1:
            q = sentinel.next[1]
            h = q.cargo[0]
            p = q.next[1]

            while p is not sentinel:
                p_cargo = p.cargo
                h_vol += h * (q.cargo[1] - p_cargo[1])
                if p_cargo[0] < h:
                    h = p_cargo[0]
                q = p
                p = q.next[1]

            h_vol += h * q.cargo[1]
            return h_vol

        else:
            p = sentinel
            q = p.prev[dim_index]

            while q.cargo != None:
                if q.ignore < dim_index:
                    q.ignore = 0
                q = q.prev[dim_index]
            q = p.prev[dim_index]

            while length > 1 and (
                    q.cargo[dim_index] > bounds[dim_index] or
                    q.prev[dim_index].cargo[dim_index] >= bounds[dim_index]
            ):
                p = q
                remove(p, dim_index, bounds)
                q = p.prev[dim_index]
                length -= 1

            q_area = q.area
            q_cargo = q.cargo
            q_prev_dim_index = q.prev[dim_index]

            if length > 1:
                h_vol = q_prev_dim_index.volume[dim_index] + q_prev_dim_index.area[dim_index] * (
                        q_cargo[dim_index] - q_prev_dim_index.cargo[dim_index])
            else:
                q_area[0] = 1
                q_area[1:dim_index + 1] = [q_area[i] * -q_cargo[i] for i in range(dim_index)]

            q.volume[dim_index] = h_vol

            if q.ignore >= dim_index:
                q_area[dim_index] = q_prev_dim_index.area[dim_index]
            else:
                q_area[dim_index] = hv_recursive(dim_index - 1, length, bounds)
                if q_area[dim_index] <= q_prev_dim_index.area[dim_index]:
                    q.ignore = dim_index

            while p is not sentinel:
                p_cargo_dim_index = p.cargo[dim_index]
                h_vol += q.area[dim_index] * (p_cargo_dim_index - q.cargo[dim_index])
                bounds[dim_index] = p_cargo_dim_index
                reinsert(p, dim_index, bounds)
                length += 1
                q = p
                p = p.next[dim_index]
                q.volume[dim_index] = h_vol

                if q.ignore >= dim_index:
                    q.area[dim_index] = q.prev[dim_index].area[dim_index]
                else:
                    q.area[dim_index] = hv_recursive(dim_index - 1, length, bounds)
                    if q.area[dim_index] <= q.prev[dim_index].area[dim_index]:
                        q.ignore = dim_index

            h_vol -= q.area[dim_index] * q.cargo[dim_index]
            return h_vol
