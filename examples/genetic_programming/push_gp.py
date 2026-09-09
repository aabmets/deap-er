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
    gp.register_gp(
        toolbox,
        pset,
        individual=creator.Individual,
        min_depth=1,
        max_depth=3,
        height_limit=8,
        select=False,
    )
    toolbox.register("evaluate", evaluate, toolbox=toolbox)

    train = tools.CaseExam.from_cases(TRAIN_CASES, N_CASES)
    held = tools.CaseExam.from_cases(HELD_CASES, N_CASES)
    pool = tools.CaseExamPool([train], held_out=held)
    return toolbox, pool, make_policy(), train.as_cases(N_CASES)


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
    stats = tools.Statistics(lambda ind: numpy.mean(ind.fitness.values))
    stats.register("min", numpy.min)
    guard = tools.PolicyActionGuard(max_promotes_per_gen=1, max_tune_gen=5)

    def decide(observation):
        return push_policy_decide(program, observation)

    _pop, logbook = tools.ea_policy(
        toolbox,
        pop,
        decide,
        generations=GENERATIONS,
        cx_prob=0.5,
        mut_prob=0.2,
        exams=pool,
        cases=cases,
        n_cases=N_CASES,
        guard=guard,
        action_kwargs={"mut_prob": 0.2},
        hof=hof,
        stats=stats,
        verbose=True,
    )
    last = logbook[-1]
    gap = last.get("generalization_gap") or {}
    print_results(hof[0], last["action"], gap.get("held_out"))


if __name__ == "__main__":
    main()
