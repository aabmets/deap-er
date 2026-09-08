import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

DIM = 5
RADIUS = 1.5
GENERATIONS = 25


def evaluate(individual):
    toward = sum((gene - 2.0) ** 2 for gene in individual)
    spread = sum((gene + 1.0) ** 2 for gene in individual)
    return toward, spread


def feasible(individual):
    return sum(gene * gene for gene in individual) <= RADIUS**2


def violation(individual):
    return max(0.0, sum(gene * gene for gene in individual) - RADIUS**2)


def setup():
    creator.create_type("FitnessMin", Fitness, weights=(-1.0, -1.0))
    creator.create_type("Individual", list, fitness=creator.FitnessMin)

    toolbox = Toolbox()
    toolbox.register("attr_float", tools.rng.uniform, -3.0, 3.0)
    toolbox.register(
        "individual",
        tools.init_repeat,
        creator.Individual,
        toolbox.attr_float,
        DIM,
    )
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("mate", tools.cx_blend, alpha=0.5)
    toolbox.register("mutate", tools.mut_gaussian, mu=0.0, sigma=0.3, mut_prob=0.2)
    toolbox.register("evaluate", evaluate)
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, (1.0e6, 1.0e6), violation))
    toolbox.register(
        "select",
        tools.sel_nsga_2,
        feasible=feasible,
        violation=violation,
    )
    toolbox.register("clone", tools.clone_individual)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min, axis=0)
    return toolbox, stats


def print_results(best_ind):
    if not feasible(best_ind):
        raise RuntimeError("Reported best individual is infeasible.")
    print(f"\nBest feasible fitness: {tuple(round(v, 4) for v in best_ind.fitness.values)}")
    print(f"Radius squared: {sum(g * g for g in best_ind):.4f} (limit {RADIUS**2:.2f})")


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=48)
    hof = tools.ParetoFront()
    tools.ea_mu_plus_lambda(
        toolbox,
        pop,
        generations=GENERATIONS,
        offsprings=48,
        survivors=48,
        cx_prob=0.6,
        mut_prob=0.3,
        hof=hof,
        stats=stats,
        verbose=True,
    )
    best = min(hof, key=lambda ind: ind.fitness.values[0]) if hof else pop[0]
    print_results(best)


if __name__ == "__main__":
    main()
