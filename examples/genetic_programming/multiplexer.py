import operator

import numpy
from deap_er import Fitness, Toolbox, creator, gp, tools

tools.rng.seed(1234)  # disables randomization

# 2-address, 4-data multiplexer (6 bits, 64 cases).
N_ADDR = 2
N_DATA = 4
N_IN = N_ADDR + N_DATA
N_CASES = 2**N_IN

CASES = [tuple((i >> b) & 1 for b in range(N_IN)) for i in range(N_CASES)]


def mux_output(bits):
    address = sum(bits[k] << k for k in range(N_ADDR))
    return bits[N_ADDR + address]


TARGETS = [mux_output(case) for case in CASES]


def if_then_else(cond, out_true, out_false):
    return out_true if cond else out_false


def evaluate(individual, toolbox):
    func = toolbox.compile(expr=individual)
    errors = sum(func(*case) != target for case, target in zip(CASES, TARGETS, strict=True))
    return (errors,)


def setup():
    pset = gp.PrimitiveSet("MAIN", N_IN)
    pset.add_primitive(operator.and_, 2)
    pset.add_primitive(operator.or_, 2)
    pset.add_primitive(operator.not_, 1)
    pset.add_primitive(if_then_else, 3)
    pset.add_terminal(1)
    pset.add_terminal(0)

    creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
    creator.create_type("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = Toolbox()
    toolbox.register("expr", gp.gen_half_and_half, prim_set=pset, min_depth=2, max_depth=4)
    toolbox.register("individual", tools.init_iterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile_tree, prim_set=pset)
    toolbox.register("evaluate", evaluate, toolbox=toolbox)
    toolbox.register("select", tools.sel_tournament, contestants=7)
    toolbox.register("mate", gp.cx_one_point)
    toolbox.register("expr_mut", gp.gen_grow, min_depth=0, max_depth=2)
    toolbox.register("mutate", gp.mut_uniform, expr=toolbox.expr_mut, prim_set=pset)
    toolbox.decorate("mate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=17))

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)
    return toolbox, stats


def print_results(best_ind):
    if best_ind.fitness.values != (0,):
        raise RuntimeError("Evolution failed to find a perfect multiplexer.")
    print("\nEvolution converged correctly.")


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=300)
    hof = tools.HallOfFame(1)
    tools.ea_simple(
        toolbox,
        pop,
        generations=40,
        cx_prob=0.8,
        mut_prob=0.1,
        hof=hof,
        stats=stats,
        verbose=True,
    )
    print_results(hof[0])


if __name__ == "__main__":
    main()
