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
from deap_er import tools
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

    dimensions = 5
    strategy = tools.Strategy(centroid=[0.0]*dimensions, sigma=1.0)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", tools.bm_sphere)
    toolbox.register("generate", strategy.generate, creator.__dict__[INDCLSNAME])
    toolbox.register("update", strategy.update)

    pop, _ = tools.ea_generate_update(toolbox, generations=100)
    best, = tools.sel_best(pop, sel_count=1)

    assert best.fitness.values < (1e-8,)

    teardown_func()


# -------------------------------------------------------------------------------------- #
def test_nsga2():
    setup_func_multi_obj()

    dimensions = 5
    bound_low, bound_up = 0.0, 1.0
    survivors = 16
    generations = 100

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform,
                     bound_low, bound_up)
    toolbox.register("individual", tools.init_repeat,
                     creator.__dict__[INDCLSNAME], toolbox.attr_float, dimensions)
    toolbox.register("population", tools.init_repeat,
                     list, toolbox.individual)

    toolbox.register("mate", tools.cx_simulated_binary_bounded,
                     low=bound_low, up=bound_up, eta=20.0)
    toolbox.register("mutate", tools.mut_polynomial_bounded,
                     low=bound_low, up=bound_up, eta=20.0, mut_prob=1.0/dimensions)

    toolbox.register("evaluate", tools.bm_zdt_1)
    toolbox.register("select", tools.sel_nsga_2)

    pop = toolbox.population(size=survivors)
    fitness = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    pop = toolbox.select(pop, len(pop))
    for gen in range(1, generations):
        offspring = tools.sel_tournament_dcd(pop, len(pop))
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

        pop = toolbox.select(pop + offspring, survivors)

    hv = tools.hypervolume(pop, [11.0, 11.0])

    assert hv > HV_THRESHOLD
    for ind in pop:
        assert not (any(numpy.asarray(ind) < bound_low) or any(numpy.asarray(ind) > bound_up))

    teardown_func()


# -------------------------------------------------------------------------------------- #
def test_mo_cma_es():
    setup_func_multi_obj_numpy()

    def distance(feasible_ind, original_ind):
        return sum((f - o)**2 for f, o in zip(feasible_ind, original_ind))

    def closest_feasible(individual):
        feasible_ind = numpy.array(individual)
        feasible_ind = numpy.maximum(bound_low, feasible_ind)
        feasible_ind = numpy.minimum(bound_up, feasible_ind)
        return feasible_ind

    def valid(individual):
        if any(individual < bound_low) or any(individual > bound_up):
            return False
        return True

    dimensions = 5
    bound_low, bound_up = 0.0, 1.0
    offsprings = 10
    survivors = 10
    generations = 500

    numpy.random.seed(128)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", tools.bm_zdt_1)
    toolbox.decorate("evaluate", tools.ClosestValidPenalty(valid, closest_feasible, 1.0e+6, distance))

    choices = numpy.random.uniform(bound_low, bound_up, (survivors, dimensions))
    population = [creator.__dict__[INDCLSNAME](x) for x in choices]
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    strategy = tools.StrategyMultiObjective(
        population, sigma=1.0, survivors=survivors, offsprings=offsprings
    )
    toolbox.register("generate", strategy.generate, creator.__dict__[INDCLSNAME])
    toolbox.register("update", strategy.update)

    for gen in range(generations):
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

    hv = tools.hypervolume(strategy.parents, [11.0, 11.0])
    assert hv > HV_THRESHOLD
    teardown_func()


# -------------------------------------------------------------------------------------- #
def test_nsga3():
    setup_func_multi_obj()

    dimensions = 5
    bound_low, bound_up = 0.0, 1.0
    survivors = 16
    generations = 100

    ref_points = tools.uniform_reference_points(2, ref_ppo=12)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform,
                     bound_low, bound_up)
    toolbox.register("individual", tools.init_repeat,
                     creator.__dict__[INDCLSNAME], toolbox.attr_float, dimensions)
    toolbox.register("population", tools.init_repeat,
                     list, toolbox.individual)

    toolbox.register("mate", tools.cx_simulated_binary_bounded,
                     low=bound_low, up=bound_up, eta=20.0)
    toolbox.register("mutate", tools.mut_polynomial_bounded,
                     low=bound_low, up=bound_up, eta=20.0, mut_prob=1.0/dimensions)
    toolbox.register("select", tools.sel_nsga_3,
                     ref_points=ref_points)

    toolbox.register("evaluate", tools.bm_zdt_1)

    pop = toolbox.population(size=survivors)
    fitness = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    pop = toolbox.select(pop, len(pop))
    for _ in range(1, generations):
        offspring = tools.var_and(toolbox, pop, 1.0, 1.0)

        invalid_ind = [ind for ind in offspring if not ind.fitness.is_valid()]

        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        pop = toolbox.select(pop + offspring, survivors)

    hv = tools.hypervolume(pop, [11.0, 11.0])

    assert hv > HV_THRESHOLD
    for ind in pop:
        assert not (any(numpy.asarray(ind) < bound_low) or any(numpy.asarray(ind) > bound_up))

    teardown_func()
