import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

N_CASES = 6
IND_SIZE = 3
TARGET = (1, 0, 1, 0, 1, 0)


def case_values(individual):
    return tuple(0.0 if individual[i % IND_SIZE] == TARGET[i] else 1.0 for i in range(N_CASES))


def evaluate(individual):
    values = case_values(individual)
    return (sum(values) / len(values),)


def solved_count(matrix):
    solved = 0
    for col in range(matrix.shape[1]):
        if numpy.any(numpy.isclose(matrix[:, col], 0.0, atol=1e-12)):
            solved += 1
    return solved


def setup():
    creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
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


def fill_archive(population, archive):
    matrix = numpy.asarray([case_values(ind) for ind in population], dtype=numpy.float64)
    basis = numpy.eye(N_CASES, dtype=numpy.float64)
    descriptors = tools.semantic_project(matrix, basis, trust_matrix=True)
    for individual, descriptor in zip(population, descriptors, strict=True):
        ranker = tools.clone_individual(individual)
        ranker.fitness.values = evaluate(individual)
        archive.add(ranker, descriptor)


def print_results(population, archive, team):
    matrix = numpy.asarray([case_values(ind) for ind in population], dtype=numpy.float64)
    best_single = max(solved_count(matrix[idx : idx + 1]) for idx in range(matrix.shape[0]))
    team_rows = numpy.asarray([case_values(member) for member in team], dtype=numpy.float64)
    team_union = solved_count(team_rows)
    if team_union <= best_single:
        raise RuntimeError(
            f"Team coverage {team_union} did not beat the best single ({best_single})."
        )
    print(f"\nArchive elites: {len(archive)}")
    print(f"Best single solved {best_single} cases; team solved {team_union}.")


def main():
    toolbox = setup()
    archive = tools.UnstructuredArchive(N_CASES, min_distance=0.2, max_elites=12)
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
    fill_archive(pop, archive)
    matrix = numpy.asarray([case_values(ind) for ind in archive], dtype=numpy.float64)
    team = tools.sel_team_archive(
        archive,
        2,
        matrix=matrix,
        trust_matrix=True,
    )
    print_results(pop, archive, team)


if __name__ == "__main__":
    main()
