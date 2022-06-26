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
from deap_er._deprecated import deprecated
import random
import math


__all__ = ['sel_spea_2', 'selSPEA2']


# ====================================================================================== #
def _partition(array, begin, end):
    x = array[begin]
    i = begin - 1
    j = end + 1
    while True:
        j -= 1
        while array[j] > x:
            j -= 1
        i += 1
        while array[i] < x:
            i += 1
        if i < j:
            array[i], array[j] = array[j], array[i]
        else:
            return j


# -------------------------------------------------------------------------------------- #
def _randomized_partition(array, begin, end):
    i = random.randint(begin, end)
    array[begin], array[i] = array[i], array[begin]
    return _partition(array, begin, end)


# -------------------------------------------------------------------------------------- #
def _randomized_select(array, begin, end, i):
    if begin == end:
        return array[begin]
    q = _randomized_partition(array, begin, end)
    k = q - begin + 1
    if i < k:
        return _randomized_select(array, begin, q, i)
    else:
        return _randomized_select(array, q + 1, end, i - k)


# -------------------------------------------------------------------------------------- #
def sel_spea_2(individuals, k):
    big_n = len(individuals)
    big_l = len(individuals[0].fitness.values)
    big_k = math.sqrt(big_n)
    strength_fits = [0] * big_n
    fits = [0] * big_n
    dominating_individuals = [list() for _ in range(big_n)]

    for i, ind_i in enumerate(individuals):
        for j, ind_j in enumerate(individuals[i+1:], i+1):
            if ind_i.fitness.dominates(ind_j.fitness):
                strength_fits[i] += 1
                dominating_individuals[j].append(i)
            elif ind_j.fitness.dominates(ind_i.fitness):
                strength_fits[j] += 1
                dominating_individuals[i].append(j)

    for i in range(big_n):
        for j in dominating_individuals[i]:
            fits[i] += strength_fits[j]

    chosen_indices = [i for i in range(big_n) if fits[i] < 1]

    if len(chosen_indices) < k:
        for i in range(big_n):
            distances = [0.0] * big_n
            for j in range(i + 1, big_n):
                dist = 0.0
                for small_l in range(big_l):
                    val = individuals[i].fitness.values[small_l] - \
                          individuals[j].fitness.values[small_l]
                    dist += val * val
                distances[j] = dist
            kth_dist = _randomized_select(distances, 0, big_n - 1, big_k)
            density = 1.0 / (kth_dist + 2.0)
            fits[i] += density

        next_indices = [(fits[i], i) for i in range(big_n) if i not in chosen_indices]
        next_indices.sort()
        chosen_indices += [i for _, i in next_indices[:k - len(chosen_indices)]]

    elif len(chosen_indices) > k:
        big_n = len(chosen_indices)
        distances = [[0.0] * big_n for _ in range(big_n)]
        sorted_indices = [[0] * big_n for _ in range(big_n)]
        for i in range(big_n):
            for j in range(i + 1, big_n):
                dist = 0.0
                for small_l in range(big_l):
                    val = individuals[chosen_indices[i]].fitness.values[small_l] - \
                          individuals[chosen_indices[j]].fitness.values[small_l]
                    dist += val * val
                distances[i][j] = dist
                distances[j][i] = dist
            distances[i][i] = -1

        for i in range(big_n):
            for j in range(1, big_n):
                small_l = j
                while small_l > 0 and distances[i][j] < distances[i][sorted_indices[i][small_l - 1]]:
                    sorted_indices[i][small_l] = sorted_indices[i][small_l - 1]
                    small_l -= 1
                sorted_indices[i][small_l] = j

        size = big_n
        to_remove = []
        while size > k:
            min_pos = 0
            for i in range(1, big_n):
                for j in range(1, size):
                    dist_i_sorted_j = distances[i][sorted_indices[i][j]]
                    dist_min_sorted_j = distances[min_pos][sorted_indices[min_pos][j]]

                    if dist_i_sorted_j < dist_min_sorted_j:
                        min_pos = i
                        break
                    elif dist_i_sorted_j > dist_min_sorted_j:
                        break

            for i in range(big_n):
                distances[i][min_pos] = float("inf")
                distances[min_pos][i] = float("inf")

                for j in range(1, size - 1):
                    if sorted_indices[i][j] == min_pos:
                        sorted_indices[i][j] = sorted_indices[i][j + 1]
                        sorted_indices[i][j + 1] = min_pos

            to_remove.append(min_pos)
            size -= 1

        for index in reversed(sorted(to_remove)):
            del chosen_indices[index]

    return [individuals[i] for i in chosen_indices]


# -------------------------------------------------------------------------------------- #
selSPEA2 = deprecated('selSPEA2', sel_spea_2)
