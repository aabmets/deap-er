import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

# Four cases; a 3-bit genome can solve at most three of them.
N_CASES = 6
IND_SIZE = 3
# Case i is solved when gene i % 3 equals TARGET[i].
TARGET = (1, 0, 1, 0, 1, 0)


def evaluate(individual):
    return tuple(0.0 if individual[i % IND_SIZE] == TARGET[i] else 1.0 for i in range(N_CASES))


def solved_count(values):
    return sum(abs(error) < 1e-12 for error in values)


def setup():
    creator.create_type("FitnessMin", Fitness, weights=(-1.0,) * N_CASES)
    creator.create_type("Individual", list, fitness=creator.FitnessMin)

    toolbox = Toolbox()
    toolbox.register("attr_bool", tools.rng.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.init_repeat,
        creator.Individual,
        toolbox.attr_bool,
        IND_SIZE,
    )
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("mate", tools.cx_two_point)
    toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.3)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("evaluate", evaluate)
    toolbox.register("clone", tools.clone_individual)
    return toolbox


def print_results(toolbox, population, team):
    tools.evaluate_invalid(toolbox, population)
    matrix = tools.fitness_case_matrix(population)
    best_single = max(solved_count(ind.fitness.values) for ind in population)
    team_union = 0
    team_matrix = tools.fitness_case_matrix(team)
    for col in range(N_CASES):
        if numpy.any(numpy.isclose(team_matrix[:, col], 0.0, atol=1e-12)):
            team_union += 1
    if team_union <= best_single:
        raise RuntimeError(
            f"Team coverage {team_union} did not beat the best single ({best_single})."
        )
    print(f"\nBest single solved {best_single} cases; team solved {team_union}.")
    print(f"Packed matrix shape: {matrix.shape}")


def main():
    toolbox = setup()
    pop = toolbox.population(size=24)
    tools.ea_simple(
        toolbox,
        pop,
        generations=8,
        cx_prob=0.5,
        mut_prob=0.3,
        verbose=True,
    )
    specialists = [creator.Individual([1, 0, 1]), creator.Individual([0, 1, 0])]
    for ind in specialists:
        ind.fitness.values = evaluate(ind)
        pop.append(ind)
    matrix = tools.fitness_case_matrix(pop)
    team = tools.sel_team(pop, 2, matrix=matrix)
    print_results(toolbox, pop, team)


if __name__ == "__main__":
    main()
