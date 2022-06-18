# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 - Mattias Aabmets and Contributors                                #
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
from deap_er.utils.deprecated import deprecated
from deap_er.tools import Logbook, Statistics, HallOfFame
from deap_er.base.toolbox import Toolbox
import random


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


# -------------------------------------------------------------------------------------- #
def ea_simple(toolbox: Toolbox,
              population: list,
              cx_prob: float,
              mut_prob: float,
              ngen: int,
              hof: HallOfFame = None,
              stats: Statistics = None,
              verbose: bool = __debug__) -> tuple[list, Logbook]:
    """
    An evolutionary algorithm. This function expects the *mate*, *mutate*,
    *select* and *evaluate* operators to be registered in the toolbox.

    :param toolbox: A Toolbox which contains the evolution operators.
    :param population: A list of individuals to vary.
    :param cx_prob: The probability of mating two individuals.
    :param mut_prob: The probability of mutating an individual.
    :param ngen: The number of generations to compute.
    :param hof: A HallOfFame object, optional.
    :param stats: A Statistics object, optional.
    :param verbose: Whether to print debug messages, optional.
    :return: Tuple of the final population and the logbook.
    """

    logbook = Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitness = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitness):
        ind.fitness.values = fit

    if hof is not None:
        hof.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = var_and(toolbox, offspring, cx_prob, mut_prob)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        if hof is not None:
            hof.update(offspring)

        population[:] = offspring

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# -------------------------------------------------------------------------------------- #
def ea_mu_plus_lambda(
        toolbox: Toolbox,
        population: list,
        ngen_ind_count: int,
        child_count: int,
        cx_prob: float,
        mut_prob: float,
        ngen: int,
        hof: HallOfFame = None,
        stats: Statistics = None,
        verbose: bool = __debug__) -> tuple[list, Logbook]:
    """
    An evolutionary algorithm. This function expects the *mate*, *mutate*,
    *select* and *evaluate* operators to be registered in the toolbox.

    :param toolbox: A Toolbox which contains the evolution operators.
    :param population: A list of individuals to vary.
    :param ngen_ind_count: The number of individuals to select for the next generation.
    :param child_count: The number of children to produce.
    :param cx_prob: The probability of mating two individuals.
    :param mut_prob: The probability of mutating an individual.
    :param ngen: The number of generations to compute.
    :param hof: A HallOfFame object, optional.
    :param stats: A Statistics object, optional.
    :param verbose: Whether to print debug messages, optional.
    :return: Tuple of the final population and the logbook.
    """

    logbook = Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitness = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitness):
        ind.fitness.values = fit

    if hof is not None:
        hof.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        offspring: list = var_or(toolbox, population, child_count, cx_prob, mut_prob)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        if hof is not None:
            hof.update(offspring)

        population[:] = toolbox.select(population + offspring, ngen_ind_count)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# -------------------------------------------------------------------------------------- #
def ea_mu_comma_lambda(toolbox: Toolbox,
                       population: list,
                       ngen_ind_count: int,
                       child_count: int,
                       cx_prob: float,
                       mut_prob: float,
                       ngen: int,
                       hof: HallOfFame = None,
                       stats: Statistics = None,
                       verbose: int = __debug__) -> tuple[list, Logbook]:
    """
    An evolutionary algorithm. This function expects the *mate*, *mutate*,
    *select* and *evaluate* operators to be registered in the toolbox.

    :param toolbox: A Toolbox which contains the evolution operators.
    :param population: A list of individuals to vary.
    :param ngen_ind_count: The number of individuals to select for the next generation.
    :param child_count: The number of children to produce.
    :param cx_prob: The probability of mating two individuals.
    :param mut_prob: The probability of mutating an individual.
    :param ngen: The number of generations to compute.
    :param hof: A HallOfFame object, optional.
    :param stats: A Statistics object, optional.
    :param verbose: Whether to print debug messages, optional.
    :return: Tuple of the final population and the logbook.
    """
    if child_count < ngen_ind_count:
        msg = 'child_count must be larger than or equal to ngen_ind_count.'
        raise ValueError(msg)

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitness = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitness):
        ind.fitness.values = fit

    if hof is not None:
        hof.update(population)

    logbook = Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        offspring = var_or(toolbox, population, child_count, cx_prob, mut_prob)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        if hof is not None:
            hof.update(offspring)

        population[:] = toolbox.select(offspring, ngen_ind_count)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# -------------------------------------------------------------------------------------- #
def ea_generate_update(toolbox: Toolbox,
                       ngen: int,
                       hof: HallOfFame = None,
                       stats: Statistics = None,
                       verbose: bool = __debug__) -> tuple[list, Logbook]:
    """
    An evolutionary algorithm. This function expects the *generate*, *update*,
    and *evaluate* operators to be registered in the toolbox.

    :param toolbox: A Toolbox which contains the evolution operators.
    :param ngen: The number of generations to compute.
    :param hof: A HallOfFame object, optional.
    :param stats: A Statistics object, optional.
    :param verbose: Whether to print debug messages, optional.
    :return: Tuple of the final population and the logbook.
    """
    logbook = Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    population = None

    for gen in range(ngen):
        population = toolbox.generate()
        fitness = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitness):
            ind.fitness.values = fit

        if hof is not None:
            hof.update(population)

        toolbox.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# ====================================================================================== #
varAnd = deprecated('varAnd', var_and)
varOr = deprecated('varOr', var_or)
eaSimple = deprecated('ea_simple', ea_simple)
eaMuPlusLambda = deprecated('eaMuPlusLambda', ea_mu_plus_lambda)
eaMuCommaLambda = deprecated('eaMuCommaLambda', ea_mu_comma_lambda)
eaGenerateUpdate = deprecated('eaGenerateUpdate', ea_generate_update)
