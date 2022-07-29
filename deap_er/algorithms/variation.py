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
from deap_er.base import Toolbox
import random


__all__ = ['var_and', 'var_or']


# ====================================================================================== #
def var_and(toolbox: Toolbox, population: list,
            cx_prob: float, mut_prob: float) -> list:
    """
    A subcomponent for evolutionary algorithms, which mates AND
    mutates each individual in the given population according to the
    given probabilities. Each of the two probabilities must be in
    the range of [0, 1]. The returned population is independent of
    the input population and has their fitness invalidated.

    :param toolbox: A Toolbox which contains the evolution operators.
    :param population: A list of individuals to evolve.
    :param cx_prob: The probability of mating two individuals.
    :param mut_prob: The probability of mutating an individual.
    :return: A list of evolved individuals.
    """
    err = "The {0} probability must be in the range of [0, 1]."
    if not (0 <= cx_prob <= 1):
        raise ValueError(err.format("crossover"))
    if not (0 <= mut_prob <= 1):
        raise ValueError(err.format("mutation"))

    offspring = [toolbox.clone(ind) for ind in population]

    for i in range(1, len(offspring), 2):
        if random.random() < cx_prob:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mut_prob:
            offspring[i], = toolbox.mutate(offspring[i])  # don't remove the comma!
            del offspring[i].fitness.values

    return offspring


# -------------------------------------------------------------------------------------- #
def var_or(toolbox: Toolbox, population: list, offsprings: int,
           cx_prob: float, mut_prob: float) -> list:
    """
    A subcomponent for evolutionary algorithms, which mates OR
    mutates each individual in the given population according to the
    given probabilities. The sum of the two probabilities must be in
    the range of [0, 1]. The returned population is independent of
    the input population and has their fitness invalidated.

    :param toolbox: A Toolbox which contains the evolution operators.
    :param population: A list of individuals to evolve.
    :param offsprings: The number of individuals to produce.
    :param cx_prob: The probability of mating two individuals.
    :param mut_prob: The probability of mutating an individual.
    :return: A list of evolved individuals.
    """
    evolve_prob = cx_prob + mut_prob
    if evolve_prob > 1.0:
        raise ValueError(
            "The sum of the crossover and the mutation "
            "probabilities must be in the range of [0, 1]."
        )

    offspring = []
    for _ in range(offsprings):
        op_choice = random.random()
        if op_choice < cx_prob:
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < evolve_prob:
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)  # don't remove the comma!
            del ind.fitness.values
            offspring.append(ind)
        else:
            offspring.append(random.choice(population))

    return offspring
