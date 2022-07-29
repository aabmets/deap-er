from deap_er import algorithms as algos
from deap_er import operators as ops
from deap_er import utilities as utils
from deap_er import records
from deap_er import creator
from deap_er import base
from deap_er import gp
import operator
import random
import numpy


random.seed(1234)  # ensure reproducibility


PARITY_FANIN_M = 6
PARITY_SIZE_M = 2 ** PARITY_FANIN_M

inputs: list = [None] * PARITY_SIZE_M
outputs: list = [None] * PARITY_SIZE_M


def fill_inputs_outputs():
    for i in range(PARITY_SIZE_M):
        inputs[i] = [None] * PARITY_FANIN_M
        value = i
        dividor = PARITY_SIZE_M
        parity = 1
        for j in range(PARITY_FANIN_M):
            dividor /= 2
            if value >= dividor:
                inputs[i][j] = 1
                parity = int(not parity)
                value -= dividor
            else:
                inputs[i][j] = 0
        outputs[i] = parity


def evaluate(individual, toolbox):
    func = toolbox.compile(expr=individual)
    result = sum(func(*in_) == out for in_, out in zip(inputs, outputs))
    return result,  # The comma is essential here.


def setup():
    pset = gp.PrimitiveSet("MAIN", PARITY_FANIN_M, "IN")
    pset.add_primitive(operator.and_, 2)
    pset.add_primitive(operator.or_, 2)
    pset.add_primitive(operator.xor, 2)
    pset.add_primitive(operator.not_, 1)
    pset.add_terminal(1)
    pset.add_terminal(0)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.gen_full, prim_set=pset, min_depth=3, max_depth=5)
    toolbox.register("individual", utils.init_iterate, creator.Individual, toolbox.expr)
    toolbox.register("population", utils.init_repeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile_tree, prim_set=pset)
    toolbox.register("evaluate", evaluate, toolbox=toolbox)
    toolbox.register("select", ops.sel_tournament, contestants=3)
    toolbox.register("mate", gp.cx_one_point)
    toolbox.register("expr_mut", gp.gen_grow, min_depth=0, max_depth=2)
    toolbox.register("mutate", gp.mut_uniform, expr=toolbox.expr_mut, prim_set=pset)

    stats = records.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    return toolbox, stats


def print_results(best_ind):
    predicted = "xor(or_(not_(1), xor(and_(1, IN2), xor(IN3, " \
                "IN0))), xor(and_(xor(xor(1, IN4), IN4), not_" \
                "(IN5)), xor(not_(IN1), xor(IN4, 1))))"
    if str(best_ind) != predicted:
        print('Symbolic regression failed to converge.')
    else:
        print(f'\nThe best individual is: \"{predicted}\".')


def main():
    fill_inputs_outputs()
    toolbox, stats = setup()
    pop = toolbox.population(size=300)
    hof = records.HallOfFame(1)

    args = dict(
        toolbox=toolbox,
        population=pop,
        generations=40,
        cx_prob=0.5,
        mut_prob=0.2,
        hof=hof,
        stats=stats,
        verbose=True
    )
    algos.ea_simple(**args)
    print_results(hof[0])


if __name__ == "__main__":
    main()

