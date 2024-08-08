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
from typing import Callable


__all__ = ['mig_ring']


# ====================================================================================== #
def mig_ring(populations: list, mig_count: int, selection: Callable,
             replacement: Callable = None, mig_indices: list = None) -> None:
    """
    Performs a ring migration between the **populations**. The migration
    first selects **mig_count** emigrants from each population using the
    specified **selection** operator and then switches the selected
    individuals between the populations.

    :param populations: A list of populations on which to operate migration.
    :param mig_count: The number of individuals to migrate.
    :param selection: The function to select emigrants from each population.
    :param replacement: The function to select which individuals will be switched.
    :param mig_indices: A list of indices indicating where the individuals from a
            particular position in the list goes. Default is a ring migration.
    :return: Nothing.
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
