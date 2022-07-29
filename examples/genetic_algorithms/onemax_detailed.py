from deap_er import operators as ops
from deap_er import utilities as utils
from deap_er import creator
from deap_er import base
import random


random.seed(1234)  # ensure reproducibility


def setup():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", utils.init_repeat, creator.Individual, toolbox.attr_bool, 100)
    toolbox.register("population", utils.init_repeat, list, toolbox.individual)
    toolbox.register("mate", ops.cx_two_point)
    toolbox.register("mutate", ops.mut_flip_bit, mut_prob=0.05)
    toolbox.register("select", ops.sel_tournament, contestants=3)
    toolbox.register("evaluate", lambda x: sum(x))

    return toolbox


def evolve(toolbox, population, max_gens, cx_prob, mut_prob):
    fitness = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitness):
        ind.fitness.values = fit
    fits = [ind.fitness.values[0] for ind in population]

    generation = 0
    while max(fits) < 100 and generation < max_gens:
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.is_valid()]
        fitness = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit
        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]
        generation += 1

    best_ind = ops.sel_best(population, sel_count=1)[0]
    return best_ind


def print_results(best_ind):
    if not all(gene == 1 for gene in best_ind):
        print('Evolution failed to converge.')
    else:
        print(f'\nThe best individual is: [1, 1, 1, ..., 1] '
              f'with a fitness score of 100.')


def main():
    toolbox = setup()
    pop = toolbox.population(size=300)

    args = dict(
        toolbox=toolbox,
        population=pop,
        max_gens=1000,
        cx_prob=0.5,
        mut_prob=0.2
    )
    best_ind = evolve(**args)
    print_results(best_ind)


if __name__ == "__main__":
    main()
