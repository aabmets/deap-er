from deap_er import utilities as utils
from deap_er import evaluators as evals
from deap_er import records
from deap_er import creator
from deap_er import base
import operator
import random
import numpy
import math


random.seed(1234)  # disables randomization


def generate(size, pmin, pmax, smin, smax):
    p_list = [random.uniform(pmin, pmax) for _ in range(size)]
    s_list = [random.uniform(smin, smax) for _ in range(size)]
    part = creator.Particle(p_list)
    part.speed = s_list
    part.smin = smin
    part.smax = smax
    return part


def update(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))


def setup():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMax,
                   speed=list, smin=None, smax=None, best=None)

    toolbox = base.Toolbox()
    toolbox.register("particle", generate, size=2, pmin=-6, pmax=6, smin=-3, smax=3)
    toolbox.register("population", utils.init_repeat, list, toolbox.particle)
    toolbox.register("update", update, phi1=2.0, phi2=2.0)
    toolbox.register("evaluate", evals.h1)

    stats = records.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    return toolbox, stats


def print_results(best_ind):
    if not best_ind.fitness.values <= (2,):
        raise RuntimeError('Evolution failed to converge.')
    print(f'\nEvolution converged correctly.')


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=5)
    logbook = records.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GENS = 1000
    best = None

    for g in range(GENS):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

    print_results(best)


if __name__ == "__main__":
    main()
