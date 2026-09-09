import math
import operator

import numpy
from deap_er import Fitness, Toolbox, creator, gp, tools

tools.rng.seed(1234)  # disables randomization

POINTS = [x / 10.0 for x in range(-10, 10)]
TARGET = numpy.asarray([x**2 for x in POINTS], dtype=numpy.float64)
N_CASES = len(POINTS)
BUDGET = 120


def safe_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def predicted_series(individual, toolbox):
    func = toolbox.compile(expr=individual)
    return numpy.asarray([func(x) for x in POINTS], dtype=numpy.float64)


def evaluate(individual, toolbox, recipe):
    predicted = predicted_series(individual, toolbox)
    ranges = [(i, i + 1) for i in recipe.train_cases]
    errors = tools.affine_case_errors(predicted, TARGET, ranges)
    full = [0.0] * recipe.n_cases
    for idx, case in enumerate(recipe.train_cases):
        full[case] = errors[idx]
    return tuple(full)


def held_out_score(individual, toolbox, recipe):
    predicted = predicted_series(individual, toolbox)
    ranges = [(i, i + 1) for i in recipe.held_cases]
    errors = tools.affine_case_errors(predicted, TARGET, ranges)
    return float(numpy.mean(errors))


def polish(individual, toolbox, recipe, remaining):
    strategy = tools.Strategy([0.0], 0.6, offsprings=4, survivors=2)

    def judge(tree):
        return (held_out_score(tree, toolbox, recipe),)

    return gp.tune_ephemerals_budget(
        individual,
        strategy,
        judge,
        n_gen=gp.MEMETIC_DEFAULT_N_GEN,
        n_evals=remaining,
        nevals_used=0,
    )


def setup(recipe):
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.sub, 2)
    pset.add_primitive(operator.mul, 2)
    pset.add_primitive(safe_div, 2)
    pset.add_ephemeral_constant("memetic_scale", lambda: tools.rng.uniform(0.5, 1.5))
    pset.rename_arguments(ARG0="x")

    creator.create_type("MemeticFit", Fitness, weights=(-1.0,) * N_CASES)
    creator.create_type("MemeticInd", gp.PrimitiveTree, fitness=creator.MemeticFit)

    toolbox = Toolbox()
    toolbox.register("expr", gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=2)
    toolbox.register("individual", tools.init_iterate, creator.MemeticInd, toolbox.expr)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile_tree, prim_set=pset)
    toolbox.register("mate", gp.cx_one_point)
    toolbox.register("expr_mut", gp.gen_full, min_depth=0, max_depth=1)
    toolbox.register("mutate", gp.mut_uniform, expr=toolbox.expr_mut, prim_set=pset)
    toolbox.register("select", recipe.make_select(downsample=8))
    toolbox.register("evaluate", evaluate, toolbox=toolbox, recipe=recipe)
    toolbox.decorate("mate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=6))
    toolbox.decorate("mutate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=6))
    return toolbox


def print_results(best_ind, toolbox, recipe, spent):
    train = [best_ind.fitness.values[i] for i in recipe.train_cases]
    train_mse = math.fsum(train) / len(train)
    held = held_out_score(best_ind, toolbox, recipe)
    if train_mse >= 0.05:
        raise RuntimeError("Memetic affine path failed to reach the train floor.")
    print(f"\nTrain affine MSE: {train_mse:.5f}")
    print(f"Held-out affine MSE: {held:.5f}")
    print(f"Memetic eval units spent: {spent}")
    print(f"Best program: {best_ind}")


def main():
    recipe = tools.case_generalization_recipe(N_CASES, fraction=0.2)
    toolbox = setup(recipe)
    pop = toolbox.population(size=50)
    tools.evaluate_invalid(toolbox, pop)
    spent = 0

    for _ in range(10):
        offspring = tools.var_and(toolbox, toolbox.select(pop, len(pop)), 0.5, 0.25)
        tools.evaluate_invalid(toolbox, offspring)
        pop[:] = offspring

    best = tools.sel_best(pop, 1)[0]
    if spent < BUDGET:
        _, cost = polish(best, toolbox, recipe, BUDGET - spent)
        spent += cost
        toolbox.evaluate(best)

    print_results(best, toolbox, recipe, spent)


if __name__ == "__main__":
    main()
