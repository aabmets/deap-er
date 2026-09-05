import itertools
import math
import operator

import numpy
from deap_er import Fitness, Toolbox, creator, tools

# Disable randomization to guarantee reproducibility
tools.rng.seed(1234)

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
    part = pclass(tools.rng.uniform(pmin, pmax) for _ in range(dim))
    part.speed = [tools.rng.uniform(smin, smax) for _ in range(dim)]
    return part


def update_particle(part, best, chi, c):
    ce1 = (c * tools.rng.uniform(0, 1) for _ in range(len(part)))
    ce2 = (c * tools.rng.uniform(0, 1) for _ in range(len(part)))
    ce1_p = map(operator.mul, ce1, map(operator.sub, best, part))
    ce2_g = map(operator.mul, ce2, map(operator.sub, part.best, part))
    a = map(
        operator.sub,
        map(operator.mul, itertools.repeat(chi), map(operator.add, ce1_p, ce2_g)),
        map(operator.mul, itertools.repeat(1 - chi), part.speed),
    )
    part.speed = list(map(operator.add, part.speed, a))
    part[:] = list(map(operator.add, part, part.speed))


def convert_swarm(swarm, rcloud, centre, dist):
    dim = len(swarm[0])
    for part in swarm:
        position = [tools.rng.gauss(0, 1) for _ in range(dim)]
        dist_ = math.sqrt(sum(x**2 for x in position))

        if dist == "gaussian":
            u = abs(tools.rng.gauss(0, 1.0 / 3.0))
            part[:] = [
                (rcloud * x * u ** (1.0 / dim) / dist_) + c
                for x, c in zip(position, centre, strict=False)
            ]
        elif dist == "uvd":
            u = tools.rng.random()
            part[:] = [
                (rcloud * x * u ** (1.0 / dim) / dist_) + c
                for x, c in zip(position, centre, strict=False)
            ]
        elif dist == "nuvd":
            u = abs(tools.rng.gauss(0, 1.0 / 3.0))
            part[:] = [(rcloud * x * u / dist_) + c for x, c in zip(position, centre, strict=False)]

        del part.fitness.values
        del part.bestfit.values
        part.best = None

    return swarm


def setup():
    creator.create_type("FitnessMax", Fitness, weights=(1.0,))
    creator.create_type(
        "Particle",
        list,
        fitness=creator.FitnessMax,
        speed=list,
        best=None,
        bestfit=creator.FitnessMax,
    )
    creator.create_type("Swarm", list, best=None, bestfit=creator.FitnessMax)

    toolbox = Toolbox()
    toolbox.register(
        "particle",
        generate_particle,
        creator.Particle,
        dim=NDIM,
        pmin=PMIN,
        pmax=PMAX,
        smin=SMIN,
        smax=SMAX,
    )
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
    if len(logbook) >= 5e5:
        raise RuntimeError("Evolution failed to converge.")
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
    print(f"\nAverage offline error: {avg_err:.3f} (<={AVG_OE_THRESHOLD}).")
    print("\nEvolution converged correctly.")


def _roaming_worst(population, rex_cl):
    worst_swarm_idx = None
    worst_swarm = None
    not_converged = 0
    for i, swarm in enumerate(population):
        for p1, p2 in itertools.combinations(swarm, 2):
            d = math.sqrt(sum((x1 - x2) ** 2.0 for x1, x2 in zip(p1, p2, strict=False)))
            if d > 2 * rex_cl:
                not_converged += 1
                if not worst_swarm or swarm.bestfit < worst_swarm.bestfit:
                    worst_swarm_idx = i
                    worst_swarm = swarm
                break
    return not_converged, worst_swarm_idx


def _resize_swarms(toolbox, population, not_converged, worst_swarm_idx):
    if not_converged == 0:
        population.append(toolbox.swarm(size=NPARTICLES))
    elif not_converged > NEXCESS:
        population.pop(worst_swarm_idx)


def _step_swarms(toolbox, population, update_fitness):
    for swarm in population:
        if swarm.best and toolbox.evaluate(swarm.best) != swarm.bestfit.values:
            swarm[:] = toolbox.convert(swarm, rcloud=RCLOUD, centre=swarm.best)
            swarm.best = None
            del swarm.bestfit.values
        for part in swarm:
            if swarm.best and part.best:
                toolbox.update(part, swarm.best)
            update_fitness(swarm, part)


def _reinit_overlapping(toolbox, population, rex_cl, update_fitness):
    reinit_swarms = set()
    for s1, s2 in itertools.combinations(range(len(population)), 2):
        if not (
            population[s1].best
            and population[s2].best
            and not (s1 in reinit_swarms or s2 in reinit_swarms)
        ):
            continue
        dist = math.sqrt(
            sum(
                (x1 - x2) ** 2.0
                for x1, x2 in zip(population[s1].best, population[s2].best, strict=False)
            )
        )
        if dist < rex_cl:
            if population[s1].bestfit <= population[s2].bestfit:
                reinit_swarms.add(s1)
            else:
                reinit_swarms.add(s2)
    for s in reinit_swarms:
        population[s] = toolbox.swarm(size=NPARTICLES)
        for part in population[s]:
            update_fitness(population[s], part)


def main():
    toolbox, stats, logbook = setup()

    def update_fitness(group, part):
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
        args = {
            "gen": ngen,
            "evals": mpb.nevals,
            "nswarm": len(population),
            "error": mpb.current_error,
            "offline_error": mpb.offline_error,
        }
        logbook.record(**args, **record)
        print(logbook.stream)

    population = [toolbox.swarm(size=NPARTICLES) for _ in range(NSWARMS)]
    for swarm in population:
        for part in swarm:
            update_fitness(swarm, part)

    log_stats()

    generation = 1
    while not stop_condition(logbook):
        rex_cl = (BOUNDS[1] - BOUNDS[0]) / (2 * len(population) ** (1.0 / NDIM))
        not_converged, worst_swarm_idx = _roaming_worst(population, rex_cl)
        _resize_swarms(toolbox, population, not_converged, worst_swarm_idx)
        _step_swarms(toolbox, population, update_fitness)
        log_stats(generation)
        _reinit_overlapping(toolbox, population, rex_cl, update_fitness)
        generation += 1


if __name__ == "__main__":
    main()
