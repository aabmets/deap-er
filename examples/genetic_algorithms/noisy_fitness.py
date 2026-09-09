import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

N_BITS = 20
POP = 40
GENERATIONS = 15


def noisy_evaluate(individual):
    signal = sum(individual)
    noise = tools.rng.uniform(-2.5, 2.5)
    return (float(signal + noise),)


def setup():
    creator.create_type("FitnessMax", Fitness, weights=(1.0,))
    creator.create_type("Individual", list, fitness=creator.FitnessMax)

    draws = {"n": 0}

    def counting_evaluate(individual):
        draws["n"] += 1
        return noisy_evaluate(individual)

    cache = tools.EvalCache(counting_evaluate, key_fn=tuple)

    def evaluate(individual):
        return cache.evaluate(individual)

    toolbox = Toolbox()
    toolbox.register("attr_bool", tools.rng.randint, 0, 1)
    toolbox.register("individual", tools.init_repeat, creator.Individual, toolbox.attr_bool, N_BITS)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("mate", tools.cx_two_point)
    toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.05)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("evaluate", evaluate)
    toolbox.register("clone", tools.clone_individual)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    return toolbox, stats, cache, draws


def print_results(best_ind, cache, draws):
    true_score = sum(best_ind)
    averaged = tools.resample(best_ind, cache.evaluate, 7, write=False)
    if averaged[0] < true_score - 1.0:
        raise RuntimeError("Resample mean diverged from the latent score.")
    race = tools.race_stop([tools.clone_individual(best_ind)], cache.evaluate, 5, min_survivors=1)
    if not race.survivors:
        raise RuntimeError("Race stop dropped the only survivor.")
    print(f"\nLatent ones: {true_score} / {N_BITS}")
    print(f"Seven-draw mean: {averaged[0]:.2f}")
    print(f"Distinct cache keys during search: {len(cache)} (evaluate calls {draws['n']})")


def main():
    toolbox, stats, cache, draws = setup()
    pop = toolbox.population(size=POP)
    hof = tools.HallOfFame(1)
    for gen in range(GENERATIONS):
        offspring = tools.var_and(toolbox, toolbox.select(pop, len(pop)), 0.5, 0.2)
        for mutant in offspring:
            tools.resample(mutant, toolbox.evaluate, 3, cache=cache, key="genes")
        hof.update(offspring)
        pop[:] = offspring
        if gen % 5 == 0:
            record = stats.compile(offspring)
            print(f"gen {gen} max {record['max']:.2f}")
    print_results(hof[0], cache, draws)


if __name__ == "__main__":
    main()
