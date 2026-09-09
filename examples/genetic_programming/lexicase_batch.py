import math
import operator

import numpy
from deap_er import Fitness, Toolbox, creator, gp, tools

tools.rng.seed(1234)  # disables randomization

POINTS = [x / 10.0 for x in range(-10, 10)]
TARGET = [x**4 + x**3 + x**2 + x for x in POINTS]
CASE_COUNT = 8


def safe_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def evaluate(individual, toolbox):
    func = toolbox.compile(expr=individual)
    return tuple((func(x) - y) ** 2 for x, y in zip(POINTS, TARGET, strict=True))


def select(individuals, sel_count):
    generation = select.generation
    select.generation += 1
    matrix = tools.fitness_case_matrix(individuals)
    phase = generation % 3
    if phase == 0:
        return tools.sel_batch_epsilon_lexicase(
            individuals,
            sel_count,
            batch_size=4,
            matrix=matrix,
        )
    if phase == 1:
        cases = tools.sample_informed_cases(individuals, CASE_COUNT, matrix=matrix)
        return tools.sel_tournament_cases(
            individuals,
            rounds=sel_count,
            contestants=3,
            cases=cases,
            matrix=matrix,
        )
    cases = tools.next_downsample_cases(
        individuals,
        CASE_COUNT,
        generation,
        mode="informed",
    )
    return tools.sel_epsilon_lexicase(
        individuals,
        sel_count,
        cases=cases,
        matrix=matrix,
        mode="epsilon_dynamic",
    )


def setup():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.sub, 2)
    pset.add_primitive(operator.mul, 2)
    pset.add_primitive(safe_div, 2)
    pset.add_primitive(operator.neg, 1)
    pset.add_primitive(math.cos, 1)
    pset.add_primitive(math.sin, 1)
    pset.add_ephemeral_constant("rand_lex_batch", lambda: tools.rng.randint(-1, 1))
    pset.rename_arguments(ARG0="x")

    creator.create_type("LexBatchFit", Fitness, weights=(-1.0,) * len(POINTS))
    creator.create_type("LexBatchInd", gp.PrimitiveTree, fitness=creator.LexBatchFit)

    toolbox = Toolbox()
    toolbox.register("expr", gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=3)
    toolbox.register("individual", tools.init_iterate, creator.LexBatchInd, toolbox.expr)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile_tree, prim_set=pset)
    toolbox.register("mate", gp.cx_one_point)
    toolbox.register("expr_mut", gp.gen_full, min_depth=0, max_depth=2)
    toolbox.register("mutate", gp.mut_uniform, expr=toolbox.expr_mut, prim_set=pset)
    select.generation = 0
    toolbox.register("select", select)
    toolbox.register("evaluate", evaluate, toolbox=toolbox)
    toolbox.decorate("mate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=8))

    stats = tools.Statistics(lambda ind: float(numpy.mean(ind.fitness.values)))
    stats.register("min", numpy.min)
    return toolbox, stats


def print_results(best_ind):
    mse = float(numpy.mean(best_ind.fitness.values))
    if mse >= 1.0:
        raise RuntimeError("Lexicase batch selectors failed to improve the seed-fixed floor.")
    print(f"\nBest mean case MSE: {mse:.4g}")
    print("Selectors used: batch ε-lexicase, tournament cases, dynamic ε")


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=100)
    hof = tools.HallOfFame(1)
    tools.ea_simple(
        toolbox,
        pop,
        generations=24,
        cx_prob=0.5,
        mut_prob=0.2,
        hof=hof,
        stats=stats,
        verbose=True,
    )
    print_results(hof[0])


if __name__ == "__main__":
    main()
