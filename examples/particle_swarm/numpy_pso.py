import math

import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

NGEN = 1000


def generate(size, pmin, pmax, smin, smax):
    p_list = numpy.array([tools.rng.uniform(pmin, pmax) for _ in range(size)])
    s_list = numpy.array([tools.rng.uniform(smin, smax) for _ in range(size)])
    part = creator.Particle(p_list)
    part.speed = s_list
    part.smin = smin
    part.smax = smax
    return part


def update(part, best, phi1, phi2):
    u1 = numpy.array([tools.rng.uniform(0, phi1) for _ in range(len(part))])
    u2 = numpy.array([tools.rng.uniform(0, phi2) for _ in range(len(part))])
    v_u1 = u1 * (part.best - part)
    v_u2 = u2 * (best - part)
    part.speed += v_u1 + v_u2
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part += part.speed


def setup():
    creator.create_type("FitnessMax", Fitness, weights=(1.0,))
    creator.create_type(
        "Particle",
        numpy.ndarray,
        fitness=creator.FitnessMax,
        speed=list,
        smin=None,
        smax=None,
        best=None,
    )

    toolbox = Toolbox()
    toolbox.register("particle", generate, size=2, pmin=-6, pmax=6, smin=-3, smax=3)
    toolbox.register("population", tools.init_repeat, list, toolbox.particle)
    toolbox.register("update", update, phi1=2.0, phi2=2.0)
    toolbox.register("evaluate", tools.bm_h1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    return toolbox, stats, logbook


def print_results(best_ind):
    if best_ind.fitness.values > (2,):
        raise RuntimeError("Evolution failed to converge.")
    print("\nEvolution converged correctly.")


def main():
    toolbox, stats, logbook = setup()
    pop = toolbox.population(size=5)

    best = None
    for g in range(NGEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if part.best is None or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if best is None or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

    print_results(best)


if __name__ == "__main__":
    main()
