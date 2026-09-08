from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

N_BITS = 30
N_DEMES = 3
DEME_SIZE = 40
GENERATIONS = 25
MIG_INTERVAL = 5


def evaluate(individual):
    return (sum(individual),)


def setup():
    creator.create_type("FitnessMax", Fitness, weights=(1.0,))
    creator.create_type("Individual", list, fitness=creator.FitnessMax)

    toolbox = Toolbox()
    toolbox.register("attr_bool", tools.rng.randint, 0, 1)
    toolbox.register("individual", tools.init_repeat, creator.Individual, toolbox.attr_bool, N_BITS)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("mate", tools.cx_two_point)
    toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.05)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("evaluate", evaluate)
    toolbox.register("clone", tools.clone_individual)

    def vary(population):
        return tools.var_and(toolbox, population, 0.5, 0.2)

    toolbox.register("vary", vary)
    return toolbox


def migrate(populations):
    tools.mig_ring(populations, mig_count=4, selection=tools.sel_best)


def print_results(best_ind):
    if not all(gene == 1 for gene in best_ind):
        raise RuntimeError("Island OneMax failed to reach all-ones.")
    print("\nEvolution converged correctly.")


def main():
    toolbox = setup()
    demes = [(toolbox, toolbox.population(size=DEME_SIZE)) for _ in range(N_DEMES)]
    for _ in range(GENERATIONS):
        tools.step_islands(demes, migrate=migrate if _ % MIG_INTERVAL == 0 else None)
    best = max((ind for _, pop in demes for ind in pop), key=lambda ind: ind.fitness.values)
    print_results(best)


if __name__ == "__main__":
    main()
