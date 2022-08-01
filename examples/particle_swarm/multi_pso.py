from deap_er import creator
from deap_er import tools
from deap_er import base
import itertools
import operator
import random
import numpy
import math


# Disable randomization to guarantee reproducibility
random.seed(1234)

# Define constants, objects and functions.
NDIM = 5
NSWARMS = 1
NPARTICLES = 5
NEXCESS = 3
RCLOUD = 0.5
SWARM_DSTRB = "nuvd"
AVG_OE_THRESHOLD = 5
AVG_OE_MEASURE_INTERVAL = 200

SCENARIO = tools.MPConfigs.ALT1
mpb = tools.MovingPeaks(dimensions=NDIM, **SCENARIO)

BOUNDS = [SCENARIO["min_coord"], SCENARIO["max_coord"]]
SMIN = -(BOUNDS[1] - BOUNDS[0]) / 2.0
SMAX = (BOUNDS[1] - BOUNDS[0]) / 2.0
PMIN = BOUNDS[0]
PMAX = BOUNDS[1]


def generate_particle(pclass, dim, pmin, pmax, smin, smax):
    part = pclass(random.uniform(pmin, pmax) for _ in range(dim)) 
    part.speed = [random.uniform(smin, smax) for _ in range(dim)]
    return part


def update_particle(part, best, chi, c):
    ce1 = (c * random.uniform(0, 1) for _ in range(len(part)))
    ce2 = (c * random.uniform(0, 1) for _ in range(len(part)))
    ce1_p = map(operator.mul, ce1, map(operator.sub, best, part))
    ce2_g = map(operator.mul, ce2, map(operator.sub, part.best, part))
    a = map(
        operator.sub, map(
            operator.mul, itertools.repeat(chi), map(
                operator.add, ce1_p, ce2_g)
        ), map(operator.mul, itertools.repeat(1 - chi), part.speed)
    )
    part.speed = list(map(operator.add, part.speed, a))
    part[:] = list(map(operator.add, part, part.speed))


def convert_swarm(swarm, rcloud, centre, dist):
    dim = len(swarm[0])
    for part in swarm:
        position = [random.gauss(0, 1) for _ in range(dim)]
        dist_ = math.sqrt(sum(x**2 for x in position))

        if dist == "gaussian":
            u = abs(random.gauss(0, 1.0/3.0))
            part[:] = [(rcloud * x * u**(1.0/dim) / dist_) + c for x, c in zip(position, centre)]
        elif dist == "uvd":
            u = random.random()
            part[:] = [(rcloud * x * u**(1.0/dim) / dist_) + c for x, c in zip(position, centre)]
        elif dist == "nuvd":
            u = abs(random.gauss(0, 1.0/3.0))
            part[:] = [(rcloud * x * u / dist_) + c for x, c in zip(position, centre)]

        del part.fitness.values
        del part.bestfit.values
        part.best = None

    return swarm


def setup():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMax,
                   speed=list, best=None, bestfit=creator.FitnessMax)
    creator.create("Swarm", list, best=None, bestfit=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("particle", generate_particle, creator.Particle,
                     dim=NDIM, pmin=PMIN, pmax=PMAX, smin=SMIN, smax=SMAX)
    toolbox.register("swarm", tools.init_repeat, creator.Swarm, toolbox.particle)
    toolbox.register("update", update_particle, chi=0.729843788, c=2.05)
    toolbox.register("convert", convert_swarm, dist=SWARM_DSTRB)
    toolbox.register("evaluate", mpb)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "nswarm", "evals", "error", "offline_error", "avg", "max"

    return toolbox, stats, logbook


def stop_condition(logbook):
    interval = AVG_OE_MEASURE_INTERVAL
    if len(logbook) >= 5e+5:
        raise RuntimeError('Evolution failed to converge.')
    elif len(logbook) % interval == 0:
        err_sum = 0
        for i in range(interval, 0, -1):
            val = logbook.select("offline_error")[-i]
            err_sum += val
        avg_err = err_sum / interval
        if avg_err <= AVG_OE_THRESHOLD:
            print_results(avg_err)
            return 1
    return 0


def print_results(avg_err):
    print(f'\nAverage offline error: {avg_err:.3f} (<={AVG_OE_THRESHOLD}).')
    print(f'\nEvolution converged correctly.')


def main():
    toolbox, stats, logbook = setup()

    def update_fitness(group):
        part.fitness.values = toolbox.evaluate(part)
        if not part.best or part.fitness > part.bestfit:
            part.best = toolbox.clone(part[:])
            part.bestfit.values = part.fitness.values
        if not group.best or part.fitness > group.bestfit:
            group.best = toolbox.clone(part[:])
            group.bestfit.values = part.fitness.values

    def log_stats(ngen=0):
        chain = itertools.chain(*population)
        record = stats.compile(chain)
        args = dict(
            gen=ngen,
            evals=mpb.nevals,
            nswarm=len(population),
            error=mpb.current_error,
            offline_error=mpb.offline_error
        )
        logbook.record(**args, **record)
        print(logbook.stream)

    # Generate the initial population.
    population = [toolbox.swarm(size=NPARTICLES) for _ in range(NSWARMS)]

    # Evaluate the initial population.
    for swarm in population:
        for part in swarm:
            update_fitness(swarm)

    log_stats()

    generation = 1

    # Define the main evolution loop.
    while not stop_condition(logbook):

        # Reset convergence variables.
        rex_cl = (BOUNDS[1] - BOUNDS[0]) / (2 * len(population) ** (1.0 / NDIM))
        worst_swarm_idx = None
        worst_swarm = None
        not_converged = 0

        # Compute the diameters of the swarms and search for the
        # worst swarm according to its best global position.
        for i, swarm in enumerate(population):
            for p1, p2 in itertools.combinations(swarm, 2):
                d = math.sqrt(sum((x1 - x2) ** 2. for x1, x2 in zip(p1, p2)))
                if d > 2 * rex_cl:
                    not_converged += 1
                    if not worst_swarm or swarm.bestfit < worst_swarm.bestfit:
                        worst_swarm_idx = i
                        worst_swarm = swarm
                    break

        # If all swarms have converged, add a swarm.
        if not_converged == 0:
            population.append(toolbox.swarm(size=NPARTICLES))

        # If too many swarms are roaming, remove the worst swarm.
        elif not_converged > NEXCESS:
            population.pop(worst_swarm_idx)

        # Update and evaluate the swarm.
        for swarm in population:

            # Check for change.
            if swarm.best and toolbox.evaluate(swarm.best) != swarm.bestfit.values:

                # Convert particles to quantum particles.
                swarm[:] = toolbox.convert(swarm, rcloud=RCLOUD, centre=swarm.best)
                swarm.best = None
                del swarm.bestfit.values

            # Not necessary to update if it's a new swarm
            # or a swarm just converted to quantum.
            for part in swarm:
                if swarm.best and part.best:
                    toolbox.update(part, swarm.best)
                update_fitness(swarm)

        log_stats(generation)

        # Apply exclusion to swarms which have the best
        # position and are not set to reinitialize.
        reinit_swarms = set()
        for s1, s2 in itertools.combinations(range(len(population)), 2):
            if (
                    population[s1].best and population[s2].best and
                    not (s1 in reinit_swarms or s2 in reinit_swarms)
            ):
                dist = 0
                for x1, x2 in zip(population[s1].best, population[s2].best):
                    dist += (x1 - x2) ** 2.
                dist = math.sqrt(dist)
                if dist < rex_cl:
                    if population[s1].bestfit <= population[s2].bestfit:
                        reinit_swarms.add(s1)
                    else:
                        reinit_swarms.add(s2)

        # Reinitialize and evaluate swarms.
        for s in reinit_swarms:
            population[s] = toolbox.swarm(size=NPARTICLES)
            for part in population[s]:
                update_fitness(population[s])

        # Update iteration counter.
        generation += 1


if __name__ == "__main__":
    main()
