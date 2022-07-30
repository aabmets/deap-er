from deap_er import algorithms as algos
from deap_er import operators as ops
from deap_er import utilities as utils
from deap_er import records
from deap_er import creator
from deap_er import base
from deap_er import gp
from functools import partial
import random
import numpy
import copy


random.seed(1234)  # disables randomization


class AntSimulator:
    direction = ["north", "east", "south", "west"]
    dir_row = [1, 0, -1, 0]
    dir_col = [0, 1, 0, -1]

    def __init__(self, max_moves):
        self.max_moves = max_moves

    def _reset(self):
        self.matrix_exc = copy.deepcopy(self.matrix)
        self.row = self.row_start
        self.col = self.col_start
        self.dir = 1
        self.moves = 0
        self.eaten = 0

    @staticmethod
    def _if_then_else(condition, out1, out2):
        out1() if condition() else out2()

    @property
    def position(self):
        return self.row, self.col, self.direction[self.dir]

    def turn_left(self):
        if self.moves < self.max_moves:
            self.moves += 1
            self.dir = (self.dir - 1) % 4

    def turn_right(self):
        if self.moves < self.max_moves:
            self.moves += 1
            self.dir = (self.dir + 1) % 4

    def move_forward(self):
        if self.moves < self.max_moves:
            self.moves += 1
            self.row = (self.row + self.dir_row[self.dir]) % self.matrix_row
            self.col = (self.col + self.dir_col[self.dir]) % self.matrix_col
            if self.matrix_exc[self.row][self.col] == "food":
                self.eaten += 1
            self.matrix_exc[self.row][self.col] = "passed"

    def sense_food(self):
        ahead_row = (self.row + self.dir_row[self.dir]) % self.matrix_row
        ahead_col = (self.col + self.dir_col[self.dir]) % self.matrix_col
        return self.matrix_exc[ahead_row][ahead_col] == "food"

    def if_food_ahead(self, out1, out2):
        return partial(self._if_then_else, self.sense_food, out1, out2)

    def run(self, routine):
        self._reset()
        while self.moves < self.max_moves:
            routine()

    def parse_matrix(self, matrix):
        self.matrix = list()
        for i, line in enumerate(matrix):
            self.matrix.append(list())
            for j, col in enumerate(line):
                if col == "#":
                    self.matrix[-1].append("food")
                elif col == ".":
                    self.matrix[-1].append("empty")
                elif col == "S":
                    self.matrix[-1].append("empty")
                    self.row_start = self.row = i
                    self.col_start = self.col = j
                    self.dir = 1
        self.matrix_row = len(self.matrix)
        self.matrix_col = len(self.matrix[0])
        self.matrix_exc = copy.deepcopy(self.matrix)


def prog(*args):
    for arg in args:
        arg()


def prog2(out1, out2):
    return partial(prog, out1, out2)


def prog3(out1, out2, out3):
    return partial(prog, out1, out2, out3)


def evaluate(individual, ant, prim_set):
    routine = gp.compile_tree(individual, prim_set)
    ant.run(routine)
    return ant.eaten,  # The comma is essential here.


def setup():
    ant = AntSimulator(max_moves=600)
    with open("art_ant_trail.txt") as trail_file:
        ant.parse_matrix(trail_file)

    pset = gp.PrimitiveSet("MAIN", 0)
    pset.add_primitive(ant.if_food_ahead, 2)
    pset.add_primitive(prog2, 2)
    pset.add_primitive(prog3, 3)
    pset.add_terminal(ant.move_forward)
    pset.add_terminal(ant.turn_left)
    pset.add_terminal(ant.turn_right)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.gen_full, prim_set=pset, min_depth=1, max_depth=2)
    toolbox.register("individual", utils.init_iterate, creator.Individual, toolbox.expr_init)
    toolbox.register("population", utils.init_repeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, ant=ant, prim_set=pset)
    toolbox.register("select", ops.sel_tournament, contestants=7)
    toolbox.register("mate", gp.cx_one_point)
    toolbox.register("expr_mut", gp.gen_full, min_depth=0, max_depth=2)
    toolbox.register("mutate", gp.mut_uniform, expr=toolbox.expr_mut, prim_set=pset)

    stats = records.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    return toolbox,  stats


def print_results(best_ind):
    if not best_ind.fitness.values > (50,):
        raise RuntimeError('Evolution failed to converge.')
    print(f'\nEvolution converged correctly.')


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=300)
    hof = records.HallOfFame(1)
    args = dict(
        toolbox=toolbox,
        population=pop,
        generations=40,
        cx_prob=0.5,
        mut_prob=0.1,
        hof=hof,
        stats=stats,
        verbose=True  # prints stats
    )
    algos.ea_simple(**args)
    print_results(hof[0])


if __name__ == "__main__":
    main()
