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
from deap_er import strategies as strats
from deap_er import evaluators as evals
from deap_er import utilities as utils
from deap_er import operators as ops
from deap_er import creator
from deap_er import base
import random
import numpy


FITCLSNAME = "FIT_TYPE"
INDCLSNAME = "IND_TYPE"

HV_THRESHOLD = 116.0  # 120.777 is the optimal value


# ====================================================================================== #
def setup_func_single_obj():
    creator.create(FITCLSNAME, base.Fitness, weights=(-1.0,))
    creator.create(INDCLSNAME, list, fitness=creator.__dict__[FITCLSNAME])


def setup_func_multi_obj():
    creator.create(FITCLSNAME, base.Fitness, weights=(-1.0, -1.0))
    creator.create(INDCLSNAME, list, fitness=creator.__dict__[FITCLSNAME])


def setup_func_multi_obj_numpy():
    creator.create(FITCLSNAME, base.Fitness, weights=(-1.0, -1.0))
    creator.create(INDCLSNAME, numpy.ndarray, fitness=creator.__dict__[FITCLSNAME])


def teardown_func():
    del creator.__dict__[FITCLSNAME]
    del creator.__dict__[INDCLSNAME]


# ====================================================================================== #
def test_standard_cma():
    setup_func_single_obj()

    NDIM = 5
    strategy = strats.Strategy(centroid=[0.0]*NDIM, sigma=1.0)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evals.sphere)
    toolbox.register("generate", strategy.generate, creator.__dict__[INDCLSNAME])
    toolbox.register("update", strategy.update)

    pop, _ = algos.ea_generate_update(toolbox, generations=100)
    best, = ops.sel_best(pop, sel_count=1)

    assert best.fitness.values < (1e-8,)

    teardown_func()


# -------------------------------------------------------------------------------------- #
def test_nsga2():
    setup_func_multi_obj()

    NDIM = 5
    BOUND_LOW, BOUND_UP = 0.0, 1.0
    MU = 16
    NGEN = 100

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, BOUND_LOW, BOUND_UP)
    toolbox.register("individual", utils.init_repeat, creator.__dict__[INDCLSNAME], toolbox.attr_float, NDIM)
    toolbox.register("population", utils.init_repeat, list, toolbox.individual)

    toolbox.register("select", ops.sel_nsga_2)
    toolbox.register("mate", ops.cx_simulated_binary_bounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    toolbox.register("mutate", ops.mut_polynomial_bounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, mut_prob=1.0/NDIM)
    toolbox.register("evaluate", evals.zdt_1)

    pop = toolbox.population(count=MU)
    fitness = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    pop = toolbox.select(pop, len(pop))
    for gen in range(1, NGEN):
        offspring = ops.sel_tournament_dcd(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= 0.9:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.is_valid()]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        pop = toolbox.select(pop + offspring, MU)

    hv = utils.hypervolume(pop, [11.0, 11.0])

    assert hv > HV_THRESHOLD

    for ind in pop:
        assert not (any(numpy.asarray(ind) < BOUND_LOW) or any(numpy.asarray(ind) > BOUND_UP))

    teardown_func()


# -------------------------------------------------------------------------------------- #
def test_mo_cma_es():
    setup_func_multi_obj_numpy()

    def distance(feasible_ind, original_ind):
        return sum((f - o)**2 for f, o in zip(feasible_ind, original_ind))

    def closest_feasible(individual):
        feasible_ind = numpy.array(individual)
        feasible_ind = numpy.maximum(BOUND_LOW, feasible_ind)
        feasible_ind = numpy.minimum(BOUND_UP, feasible_ind)
        return feasible_ind

    def valid(individual):
        if any(individual < BOUND_LOW) or any(individual > BOUND_UP):
            return False
        return True

    NDIM = 5
    BOUND_LOW, BOUND_UP = 0.0, 1.0
    MU, LAMBDA = 10, 10
    NGEN = 500

    numpy.random.seed(128)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evals.zdt_1)
    toolbox.decorate("evaluate", utils.ClosestValidPenalty(valid, closest_feasible, 1.0e+6, distance))

    population = [creator.__dict__[INDCLSNAME](x) for x in numpy.random.uniform(BOUND_LOW, BOUND_UP, (MU, NDIM))]
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    strategy = strats.StrategyMultiObjective(population, sigma=1.0, mu=MU, lambda_=LAMBDA)

    toolbox.register("generate", strategy.generate, creator.__dict__[INDCLSNAME])
    toolbox.register("update", strategy.update)

    for gen in range(NGEN):
        population = toolbox.generate()

        fitness = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitness):
            ind.fitness.values = fit

        toolbox.update(population)

    num_valid = 0
    for ind in strategy.parents:
        dist = distance(closest_feasible(ind), ind)
        if numpy.isclose(dist, 0.0, rtol=1.e-5, atol=1.e-5):
            num_valid += 1
    assert num_valid >= len(strategy.parents)

    hv = utils.hypervolume(strategy.parents, [11.0, 11.0])
    assert hv > HV_THRESHOLD
    teardown_func()


# -------------------------------------------------------------------------------------- #
def test_nsga3():
    setup_func_multi_obj()
    NDIM = 5
    BOUND_LOW, BOUND_UP = 0.0, 1.0
    MU = 16
    NGEN = 100

    ref_points = ops.uniform_reference_points(2, ppo=12)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, BOUND_LOW, BOUND_UP)
    toolbox.register("individual", utils.init_repeat, creator.__dict__[INDCLSNAME], toolbox.attr_float, NDIM)
    toolbox.register("population", utils.init_repeat, list, toolbox.individual)

    toolbox.register("evaluate", evals.zdt_1)
    toolbox.register("mate", ops.cx_simulated_binary_bounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    toolbox.register("mutate", ops.mut_polynomial_bounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, mut_prob=1.0/NDIM)
    toolbox.register("select", ops.sel_nsga_3, ref_points=ref_points)

    pop = toolbox.population(count=MU)
    fitness = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    pop = toolbox.select(pop, len(pop))
    for _ in range(1, NGEN):
        offspring = algos.var_and(toolbox, pop, 1.0, 1.0)

        invalid_ind = [ind for ind in offspring if not ind.fitness.is_valid()]

        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        pop = toolbox.select(pop + offspring, MU)

    hv = utils.hypervolume(pop, [11.0, 11.0])
    assert hv > HV_THRESHOLD
    for ind in pop:
        assert not (any(numpy.asarray(ind) < BOUND_LOW) or any(numpy.asarray(ind) > BOUND_UP))

    teardown_func()
