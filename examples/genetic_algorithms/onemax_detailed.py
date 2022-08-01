from deap_er import creator
from deap_er import tools
from deap_er import base
import random


random.seed(1234)  # disables randomization

NGEN = 1000
CX_PROB = 0.5
MUT_PROB = 0.2


def setup():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.init_repeat, creator.Individual, toolbox.attr_bool, 100)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("mate", tools.cx_two_point)
    toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.05)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("evaluate", lambda x: sum(x))

    return toolbox


def print_results(best_ind):
    if not all(gene == 1 for gene in best_ind):
        raise RuntimeError('Evolution failed to converge.')
    print(f'\nEvolution converged correctly.')


def main():
    toolbox = setup()
    population = toolbox.population(size=300)

    fitness = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitness):
        ind.fitness.values = fit
    fits = [ind.fitness.values[0] for ind in population]

    generation = 0
    while max(fits) < 100 and generation < NGEN:
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.is_valid()]
        fitness = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit
        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]
        generation += 1

    best_ind = tools.sel_best(population, sel_count=1)[0]
    print_results(best_ind)


if __name__ == "__main__":
    main()
