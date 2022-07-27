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
from deap_er import algorithms as algos
from deap_er import operators as ops
from deap_er import utilities as utils
from deap_er import records
from deap_er import creator
from deap_er import base
import random
import numpy
import array


random.seed(64)


# -------------------------------------------------------- #
def eval_one_max(individual):
    return sum(individual)


# -------------------------------------------------------- #
def main():
    toolbox = base.Toolbox()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", utils.init_repeat, creator.Individual, toolbox.attr_bool, 100)
    toolbox.register("population", utils.init_repeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_one_max)
    toolbox.register("mate", ops.cx_two_point)
    toolbox.register("mutate", ops.mut_flip_bit, mut_prob=0.05)
    toolbox.register("select", ops.sel_tournament, contestants=3)

    stats = records.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop = toolbox.population(count=300)
    hof = records.HallOfFame(1)

    args = dict(
        toolbox=toolbox,
        population=pop,
        generations=40,
        cx_prob=0.5,
        mut_prob=0.2,
        hof=hof,
        stats=stats,
        verbose=True
    )
    pop, log = algos.ea_simple(**args)
    return pop, log, hof


# -------------------------------------------------------- #
if __name__ == "__main__":
    main()
