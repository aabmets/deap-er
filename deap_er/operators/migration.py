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
from deap_er._datatypes import SetItemSeq
from typing import Callable


__all__ = ['mig_ring', 'migRing']


# ====================================================================================== #
def mig_ring(populations: SetItemSeq, mig_count: int, selection: Callable,
             replacement: Callable = None, mig_indices: SetItemSeq = None) -> None:
    """
    Performs a ring migration between the *populations*. The migration
    first selects *mig_count* emigrants from each population using the
    specified *selection* operator and then switches the selected
    individuals between the populations.

    :param populations: A list of (sub-)populations on which to operate migration.
    :param mig_count: The number of individuals to migrate.
    :param selection: The function to select emigrants from each population.
    :param replacement: The function to select which individuals will be switched.
    :param mig_indices: A list of indices indicating where the individuals from
        a particular position in the list goes. This defaults to a ring migration.
    :returns: None
    """
    nbr_demes = len(populations)
    if mig_indices is None:
        mig_indices = list(range(1, nbr_demes)) + [0]

    immigrants = [[] for _ in range(nbr_demes)]
    emigrants = [[] for _ in range(nbr_demes)]

    for from_deme in range(nbr_demes):
        emigrants[from_deme].extend(selection(populations[from_deme], mig_count))
        if replacement is None:
            immigrants[from_deme] = emigrants[from_deme]
        else:
            emigrants = replacement(populations[from_deme], mig_count)
            immigrants[from_deme].extend(emigrants)

    for from_deme, to_deme in enumerate(mig_indices):
        for i, immigrant in enumerate(immigrants[to_deme]):
            indx = populations[to_deme].index(immigrant)
            populations[to_deme][indx] = emigrants[from_deme][i]


# -------------------------------------------------------------------------------------- #
migRing = deprecated('migRing', mig_ring)
