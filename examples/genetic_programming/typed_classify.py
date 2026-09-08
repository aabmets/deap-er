import operator

import numpy
from deap_er import Fitness, Toolbox, creator, gp, tools

tools.rng.seed(1234)  # disables randomization

# In-script strongly typed table: label is True when x > 0.5.
ROWS = [(x / 10.0, x / 10.0 > 0.5) for x in range(11)]


def if_then_else(cond, out_true, out_false):
    return out_true if cond else out_false


def evaluate(individual, toolbox):
    func = toolbox.compile(expr=individual)
    correct = sum(bool(func(x)) is label for x, label in ROWS)
    return (correct / len(ROWS),)


def setup():
    pset = gp.PrimitiveSetTyped("MAIN", [float], bool)
    pset.add_primitive(operator.gt, [float, float], bool)
    pset.add_primitive(operator.lt, [float, float], bool)
    pset.add_primitive(operator.and_, [bool, bool], bool)
    pset.add_primitive(operator.or_, [bool, bool], bool)
    pset.add_primitive(operator.not_, [bool], bool)
    pset.add_primitive(if_then_else, [bool, bool, bool], bool)
    pset.add_terminal(0.5, float)
    pset.add_terminal(True, bool)
    pset.add_terminal(False, bool)
    pset.rename_arguments(ARG0="x")

    creator.create_type("FitnessMax", Fitness, weights=(1.0,))
    creator.create_type("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = Toolbox()
    toolbox.register("expr", gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=3)
    toolbox.register("individual", tools.init_iterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile_tree, prim_set=pset)
    toolbox.register("evaluate", evaluate, toolbox=toolbox)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("mate", gp.cx_one_point)
    toolbox.register("expr_mut", gp.gen_full, min_depth=0, max_depth=2)
    toolbox.register("mutate", gp.mut_uniform, expr=toolbox.expr_mut, prim_set=pset)
    toolbox.decorate("mate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=8))

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)
    return toolbox, stats


def print_results(best_ind):
    if best_ind.fitness.values[0] < 0.9:
        raise RuntimeError("Training accuracy is below the seed-fixed floor.")
    print(f"\nTraining accuracy: {best_ind.fitness.values[0]:.2f}")


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=80)
    hof = tools.HallOfFame(1)
    tools.ea_simple(
        toolbox,
        pop,
        generations=15,
        cx_prob=0.5,
        mut_prob=0.2,
        hof=hof,
        stats=stats,
        verbose=True,
    )
    print_results(hof[0])


if __name__ == "__main__":
    main()
