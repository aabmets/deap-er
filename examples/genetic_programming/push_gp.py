import math
import operator

import numpy
from deap_er import Fitness, Toolbox, creator, gp, tools
from deap_er.private.programming.policy_loop import policy_action_index
from deap_er.private.programming.policy_push import (
    EMIT,
    GT,
    LOAD_UNSOLVED,
    PUSH_INT,
    PushPolicyProgram,
    push_policy_decide,
)

tools.rng.seed(1234)

POINTS = [x / 10.0 for x in range(-10, 10)]
TARGET = [x**4 + x**3 + x**2 + x for x in POINTS]
N_CASES = len(POINTS)
TRAIN_CASES = list(range(N_CASES - 4))
HELD_CASES = list(range(N_CASES - 4, N_CASES))
GENERATIONS = 12
POP_SIZE = 80


def two():
    return 2


def safe_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def evaluate(individual, toolbox):
    func = toolbox.compile(expr=individual)
    return tuple((func(x) - y) ** 2 for x, y in zip(POINTS, TARGET, strict=True))


def make_policy():
    # If the observed tape still has more than one unsolved case, ask
    # for a new lexicase exam. Otherwise skip. Default is skip_promote.
    return PushPolicyProgram(
        code=(
            LOAD_UNSOLVED,
            PUSH_INT,
            1,
            GT,
            EMIT,
            policy_action_index("next_lexicase_cases"),
        ),
        default_action=policy_action_index(tools.POLICY_ACTION_SKIP_PROMOTE),
    )


def setup():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.add_primitive(operator.add, 2, weight=2.0)
    pset.add_primitive(operator.sub, 2, weight=2.0)
    pset.add_primitive(operator.mul, 2, weight=2.0)
    pset.add_primitive(safe_div, 2, weight=0.5)
    pset.add_primitive(operator.neg, 1, weight=0.5)
    pset.add_primitive(math.cos, 1, weight=0.25)
    pset.add_primitive(math.sin, 1, weight=0.25)
    pset.add_terminal(two, call_zero=True)
    pset.add_ephemeral_constant("rand101", lambda: tools.rng.randint(-1, 1))
    pset.rename_arguments(ARG0="x")

    creator.create_type("FitnessMin", Fitness, weights=(-1.0,) * N_CASES)
    creator.create_type("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = Toolbox()
    toolbox.register("expr", gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=3)
    toolbox.register("individual", tools.init_iterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("clone", tools.clone_individual)
    toolbox.register("compile", gp.compile_tree, prim_set=pset)
    toolbox.register("mate", gp.cx_one_point)
    toolbox.register("expr_mut", gp.gen_full, min_depth=0, max_depth=2)
    toolbox.register("mutate", gp.mut_uniform, expr=toolbox.expr_mut, prim_set=pset)
    toolbox.register("evaluate", evaluate, toolbox=toolbox)
    toolbox.decorate("mate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=8))

    train = tools.CaseExam.from_cases(TRAIN_CASES, N_CASES)
    held = tools.CaseExam.from_cases(HELD_CASES, N_CASES)
    pool = tools.CaseExamPool([train], held_out=held)
    return toolbox, pool, make_policy(), train.as_cases(N_CASES)


def step_policy(program, pool, elites, nevals, rejected, guard):
    solve_bits = tools.policy_solve_bits_from_fitness(elites[0].fitness.values)
    train_score, held_out_score = tools.policy_exam_scores(pool, elites)
    observation = tools.policy_observe(
        solve_bits=solve_bits,
        train_score=train_score,
        held_out_score=held_out_score,
        nevals=nevals,
        last_action_rejected=rejected,
    )
    action = push_policy_decide(program, observation)
    result = tools.apply_policy_action(
        action,
        exams=pool,
        elites=elites,
        matrix=tools.fitness_case_matrix(elites),
        mut_prob=0.2,
        guard=guard,
    )
    return action, result, train_score, held_out_score


def print_results(best_ind, action, held_out_score):
    mse = float(numpy.mean(best_ind.fitness.values))
    print(f"\nBest program: {best_ind}")
    print(f"Infix: {gp.tree_to_infix(best_ind)}")
    print(f"Mean squared error: {mse:.6g}")
    print(f"Last policy action: {action}")
    print(f"Held-out exam score: {held_out_score}")


def main():
    toolbox, pool, program, cases = setup()
    pop = toolbox.population(size=POP_SIZE)
    hof = tools.HallOfFame(1)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "action", "held_out", "min"]
    guard = tools.PolicyActionGuard(max_promotes_per_gen=1, max_tune_gen=5)
    used = tools.evaluate_invalid(toolbox, pop)
    hof.update(pop)
    rejected = False
    action = tools.POLICY_ACTION_SKIP_PROMOTE
    held_out_score = None

    for gen in range(GENERATIONS):
        guard.begin_generation()
        elites = tools.sel_best(pop, sel_count=8)
        action, result, train_score, held_out_score = step_policy(
            program, pool, elites, used, rejected, guard
        )
        rejected = result.rejected
        if result.applied and action == "next_lexicase_cases":
            cases = result.value
        selected = tools.sel_lexicase(
            pop, len(pop), cases=cases, matrix=tools.fitness_case_matrix(pop)
        )
        pop[:] = tools.var_and(toolbox, selected, 0.5, 0.2)
        used += tools.evaluate_invalid(toolbox, pop)
        hof.update(pop)
        min_fit = float(numpy.mean(hof[0].fitness.values))
        logbook.record(
            gen=gen,
            nevals=used,
            action=action,
            held_out=held_out_score,
            min=min_fit,
            generalization_gap=tools.policy_generalization_gap(train_score, held_out_score),
        )
        print(logbook.stream)

    print_results(hof[0], action, held_out_score)


if __name__ == "__main__":
    main()
