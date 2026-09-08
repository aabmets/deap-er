import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

N_BITS = 24
POP = 30
N_EVALS = 90
GENERATION_CAP = 40


def setup():
    creator.create_type("FitnessMax", Fitness, weights=(1.0,))
    creator.create_type("Individual", list, fitness=creator.FitnessMax)

    misses = {"n": 0}

    def raw_evaluate(individual):
        misses["n"] += 1
        return (sum(individual),)

    cache = tools.EvalCache(raw_evaluate, key_fn=tuple)
    calls = {"n": 0}

    def cached_evaluate(individual):
        calls["n"] += 1
        return cache.evaluate(individual)

    toolbox = Toolbox()
    toolbox.register("attr_bool", tools.rng.randint, 0, 1)
    toolbox.register("individual", tools.init_repeat, creator.Individual, toolbox.attr_bool, N_BITS)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("mate", tools.cx_two_point)
    toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.05)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("evaluate", cached_evaluate)
    toolbox.register("clone", tools.clone_individual)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    return toolbox, stats, cache, calls, misses


def print_results(logbook, calls, misses):
    hits = calls["n"] - misses["n"]
    if hits <= 0:
        raise RuntimeError("EvalCache recorded no hits.")
    gens = logbook.select("gen")
    nevals = logbook.select("nevals")
    if gens[-1] >= GENERATION_CAP:
        raise RuntimeError("Loop did not stop on the evaluation budget.")
    if sum(nevals) < N_EVALS:
        raise RuntimeError("Recorded evaluations did not meet the budget.")
    print(f"\nCache hits: {hits} (calls {calls['n']}, misses {misses['n']})")
    print(f"Stopped at generation {gens[-1]} after {sum(nevals)} assignments.")


def main():
    toolbox, stats, cache, calls, misses = setup()
    pop = toolbox.population(size=POP)
    hof = tools.HallOfFame(1)
    _pop, log = tools.ea_simple(
        toolbox,
        pop,
        generations=GENERATION_CAP,
        cx_prob=0.5,
        mut_prob=0.2,
        hof=hof,
        stats=stats,
        verbose=True,
        n_evals=N_EVALS,
    )
    print_results(log, calls, misses)
    print(f"Unique cached expressions: {len(cache)}")


if __name__ == "__main__":
    main()
