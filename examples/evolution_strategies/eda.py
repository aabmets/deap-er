import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

DIM = 10
POP = 40
SELECTED = 12
GENERATIONS = 25


class DiagonalGaussian:
    """Sample a diagonal Gaussian whose mean and variance track elites."""

    def __init__(self, dim):
        self.mean = numpy.zeros(dim)
        self.std = numpy.ones(dim)

    def generate(self, ind_init):
        samples = [tools.rng.gauss(self.mean[i], self.std[i]) for i in range(len(self.mean))]
        return [ind_init(samples)]

    def generate_pop(self, ind_init, count):
        return [self.generate(ind_init)[0] for _ in range(count)]

    def update(self, population):
        elite = tools.sel_best(population, SELECTED)
        matrix = numpy.asarray(elite, dtype=float)
        self.mean = matrix.mean(axis=0)
        self.std = numpy.maximum(matrix.std(axis=0), 1e-6)


def setup():
    creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
    creator.create_type("Individual", list, fitness=creator.FitnessMin)
    strategy = DiagonalGaussian(DIM)
    toolbox = Toolbox()
    toolbox.register("evaluate", tools.bm_sphere)
    toolbox.register("generate", strategy.generate_pop, creator.Individual, POP)
    toolbox.register("update", strategy.update)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)
    return toolbox, stats


def print_results(best_ind):
    if best_ind.fitness.values[0] >= 1.0:
        raise RuntimeError("EDA failed to reach the seed-fixed floor.")
    print(f"\nBest sphere fitness: {best_ind.fitness.values[0]:.6f}")


def main():
    toolbox, stats = setup()
    hof = tools.HallOfFame(1)
    tools.ea_generate_update(
        toolbox,
        generations=GENERATIONS,
        hof=hof,
        stats=stats,
        verbose=True,
    )
    print_results(hof[0])


if __name__ == "__main__":
    main()
