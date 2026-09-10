import operator

import numpy
from deap_er import Fitness, Toolbox, creator, gp, tools

tools.rng.seed(1234)  # disables randomization

N_CASES = 4


def evaluate(individual, toolbox):
    func = toolbox.compile(expr=individual)
    return tuple((func(x) - x) ** 2 for x in (0.0, 0.5, 1.0, 1.5))


def select(individuals, sel_count, pset):
    matrix = tools.fitness_case_matrix(individuals)
    structural = tools.structural_meta_case_columns(
        individuals,
        prim_set=pset,
        columns=("size", "depth"),
    )
    trusted = numpy.hstack([matrix, structural])
    weights = (-1.0,) * matrix.shape[1] + tools.structural_meta_case_weights(("size", "depth"))
    return tools.sel_lexicase(
        individuals,
        sel_count,
        cases=[0],
        matrix=trusted,
        trust_matrix=True,
        fit_weights=weights,
    )


def setup():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.mul, 2)
    pset.add_ephemeral_constant("rand101", lambda: tools.rng.randint(-1, 1))
    pset.rename_arguments(ARG0="x")

    creator.create_type("FitnessMin", Fitness, weights=(-1.0,) * N_CASES)
    creator.create_type("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = Toolbox()
    toolbox.register("expr", gp.gen_grow, prim_set=pset, min_depth=1, max_depth=3)
    toolbox.register("individual", tools.init_iterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile_tree, prim_set=pset)
    toolbox.register("mate", gp.cx_one_point)
    toolbox.register("expr_mut", gp.gen_full, min_depth=0, max_depth=2)
    toolbox.register("mutate", gp.mut_uniform, expr=toolbox.expr_mut, prim_set=pset)
    toolbox.register("select", select, pset=pset)
    toolbox.register("evaluate", evaluate, toolbox=toolbox)
    toolbox.decorate("mate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=6))
    toolbox.decorate("mutate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=6))
    return toolbox


def print_results(best_ind):
    if len(best_ind) > 5:
        raise RuntimeError("Structural meta-case did not prefer a compact tree.")
    print(f"\nBest tree size: {len(best_ind)} nodes")
    print(f"Best program: {best_ind}")


def main():
    toolbox = setup()
    pop = toolbox.population(size=40)
    tools.evaluate_invalid(toolbox, pop)
    for _ in range(12):
        offspring = tools.var_and(toolbox, toolbox.select(pop, len(pop)), 0.5, 0.3)
        tools.evaluate_invalid(toolbox, offspring)
        pop[:] = offspring
    best = tools.sel_best(pop, 1)[0]
    print_results(best)


if __name__ == "__main__":
    main()
