import math
import operator

import numpy
from deap_er import Fitness, Toolbox, creator, gp, tools

tools.rng.seed(1234)  # disables randomization

POINTS = [x / 10.0 for x in range(-10, 10)]
TARGET = [x**2 for x in POINTS]
N_CASES = len(POINTS)


def safe_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def evaluate(individual, toolbox):
    func = toolbox.compile(expr=individual)
    return tuple((func(x) - y) ** 2 for x, y in zip(POINTS, TARGET, strict=True))


def setup(recipe):
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.sub, 2)
    pset.add_primitive(operator.mul, 2)
    pset.add_primitive(safe_div, 2)
    pset.add_ephemeral_constant("rand101", lambda: tools.rng.randint(-1, 1))
    pset.rename_arguments(ARG0="x")

    creator.create_type("FitnessMin", Fitness, weights=(-1.0,) * N_CASES)
    creator.create_type("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = Toolbox()
    toolbox.register("expr", gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=2)
    toolbox.register("individual", tools.init_iterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile_tree, prim_set=pset)
    toolbox.register("mate", gp.cx_one_point)
    toolbox.register("expr_mut", gp.gen_full, min_depth=0, max_depth=2)
    toolbox.register("mutate", gp.mut_uniform, expr=toolbox.expr_mut, prim_set=pset)
    toolbox.register("select", recipe.make_select(downsample=6))
    toolbox.register("evaluate", evaluate, toolbox=toolbox)
    toolbox.decorate("mate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=8))
    return toolbox


def score_held_out(individual, toolbox, recipe):
    cases = recipe.held_cases
    values = toolbox.evaluate(individual)
    return math.fsum(values[i] for i in cases) / len(cases)


def print_results(best_ind, toolbox, recipe):
    held = score_held_out(best_ind, toolbox, recipe)
    train = [best_ind.fitness.values[i] for i in recipe.train_cases]
    train_mse = math.fsum(train) / len(train)
    if train_mse >= 0.5:
        raise RuntimeError("Generalization path failed to improve train MSE.")
    print(f"\nTrain MSE: {train_mse:.4f}")
    print(f"Held-out MSE: {held:.4f}")


def main():
    recipe = tools.case_generalization_recipe(N_CASES, fraction=0.2)
    toolbox = setup(recipe)
    pop = toolbox.population(size=60)
    tools.evaluate_invalid(toolbox, pop)
    nevals = 0

    def evaluate_cases(individual, cases):
        values = toolbox.evaluate(individual)
        return [values[i] for i in cases]

    for _gen in range(10):
        offspring = tools.var_and(toolbox, toolbox.select(pop, len(pop)), 0.5, 0.2)
        halving = tools.evaluate_case_halving(
            offspring,
            evaluate_cases,
            recipe.train_cases,
            n_cases=N_CASES,
            eta=2,
            min_cases=2,
        )
        nevals += halving.nevals
        pop[:] = halving.survivors
    best = tools.sel_best(pop, 1)[0]
    print_results(best, toolbox, recipe)
    print(f"Case-eval units spent: {nevals}")


if __name__ == "__main__":
    main()
