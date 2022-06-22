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
from deap_er.base.toolbox import Toolbox
import random


__all__ = ['var_and', 'varAnd', 'var_or', 'varOr']


# ====================================================================================== #
def var_and(toolbox: Toolbox,
            population: list,
            cx_prob: float,
            mut_prob: float) -> list:
    """
    Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.

    :param toolbox: A Toolbox which contains the evolution operators.
    :param population: A list of individuals to vary.
    :param cx_prob: The probability of mating two individuals.
    :param mut_prob: The probability of mutating an individual.
    :return: A list of varied individuals.
    """
    offspring = [toolbox.clone(ind) for ind in population]

    for i in range(1, len(offspring), 2):
        if random.random() < cx_prob:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mut_prob:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


# -------------------------------------------------------------------------------------- #
def var_or(toolbox: Toolbox,
           population: list,
           child_count: int,
           cx_prob: float,
           mut_prob: float) -> list:
    """
    Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.

    :param toolbox: A Toolbox which contains the evolution operators.
    :param population: A list of individuals to vary.
    :param child_count: The number of children to produce.
    :param cx_prob: The probability of mating two individuals.
    :param mut_prob: The probability of mutating an individual.
    :return: A list of varied individuals.
    """
    offspring = []
    total_prob = cx_prob + mut_prob
    for _ in range(child_count):
        op_choice = random.uniform(0, total_prob)
        if op_choice < cx_prob:
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < total_prob:
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:
            offspring.append(random.choice(population))

    return offspring


# ====================================================================================== #
varAnd = deprecated('varAnd', var_and)
varOr = deprecated('varOr', var_or)
