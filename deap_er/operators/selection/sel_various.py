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
from operator import attrgetter
import random


__all__ = [
    'sel_random', 'sel_best', 'sel_worst', 'sel_roulette',
    'sel_stochastic_universal_sampling'
]


# ====================================================================================== #
def sel_random(individuals: list, sel_count: int) -> list:
    """
    Selects randomly **sel_count** individuals from the input **individuals**.

    :param individuals: A list of individuals to select from.
    :param sel_count: The number of individuals to select.
    :return: A list of selected individuals.
    """
    return [random.choice(individuals) for _ in range(sel_count)]


# -------------------------------------------------------------------------------------- #
def sel_best(individuals: list, sel_count: int,
             fit_attr: str = "fitness") -> list:
    """
    Selects the best **sel_count** individuals from the input **individuals**.

    :param individuals: A list of individuals to select from.
    :param sel_count: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as the selection criterion.
    :return: A list of selected individuals.
    """
    key = attrgetter(fit_attr)
    return sorted(individuals, key=key, reverse=True)[:sel_count]


# -------------------------------------------------------------------------------------- #
def sel_worst(individuals: list, sel_count: int,
              fit_attr: str = "fitness") -> list:
    """
    Selects the worst **sel_count** individuals among the input **individuals**.

    :param individuals: A list of individuals to select from.
    :param sel_count: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as the selection criterion.
    :return: A list of selected individuals.
    """
    key = attrgetter(fit_attr)
    return sorted(individuals, key=key)[:sel_count]


# -------------------------------------------------------------------------------------- #
def sel_roulette(individuals: list, sel_count: int,
                 fit_attr: str = "fitness") -> list:
    """
    Selects **sel_count** individuals from the input **individuals** using
    **sel_count** spins of a roulette. The selection is made by looking
    only at the first objective of each individual. The returned list
    contains references to the input **individuals**.

    :param individuals: A list of individuals to select from.
    :param sel_count: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as the selection criterion.
    :return: A list of selected individuals.
    """
    key = attrgetter(fit_attr)
    sorted_ = sorted(individuals, key=key, reverse=True)
    sum_fits = sum(getattr(ind, fit_attr).values[0] for ind in individuals)
    chosen = []
    for _ in range(sel_count):
        u = random.random() * sum_fits
        sum_ = 0
        for ind in sorted_:
            sum_ += getattr(ind, fit_attr).values[0]
            if sum_ > u:
                chosen.append(ind)
                break

    return chosen


# -------------------------------------------------------------------------------------- #
def sel_stochastic_universal_sampling(individuals: list, sel_count: int,
                                      fit_attr: str = "fitness") -> list:
    """
    Selects the **sel_count** individuals among the input **individuals**.
    The selection is made by using a single random value to sample all the
    individuals by choosing them at evenly spaced intervals. The returned
    list contains references to the input **individuals**.

    :param individuals: A list of individuals to select from.
    :param sel_count: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as the selection criterion.
    :return: A list of selected individuals.
    """
    key = attrgetter(fit_attr)
    sorted_ = sorted(individuals, key=key, reverse=True)
    sum_fits = sum(getattr(ind, fit_attr).values[0] for ind in individuals)

    distance = sum_fits / float(sel_count)
    start = random.uniform(0, distance)
    points = [start + i * distance for i in range(sel_count)]

    chosen = []
    for p in points:
        i = 0
        sum_ = getattr(sorted_[i], fit_attr).values[0]
        while sum_ < p:
            i += 1
            sum_ += getattr(sorted_[i], fit_attr).values[0]
        chosen.append(sorted_[i])

    return chosen
