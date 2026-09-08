import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

N_BITS = 20


def setup():
    creator.create_type("FitnessMax", Fitness, weights=(1.0,))
    creator.create_type("Individual", list, fitness=creator.FitnessMax)

    toolbox = Toolbox()
    toolbox.register("attr_bool", tools.rng.randint, 0, 1)
    toolbox.register("individual", tools.init_repeat, creator.Individual, toolbox.attr_bool, N_BITS)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("mate", tools.cx_two_point)
    toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.05)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("evaluate", lambda ind: (sum(ind),))
    toolbox.register("clone", tools.clone_individual)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    return toolbox, stats


def print_results(history, best_ind):
    if not history.genealogy_tree:
        raise RuntimeError("Genealogy tree is empty.")
    has_parents = any(parents for parents in history.genealogy_tree.values())
    if not has_parents:
        raise RuntimeError("Genealogy recorded no parent links.")
    tree = history.get_genealogy(best_ind)
    if not tree:
        raise RuntimeError("get_genealogy returned an empty graph.")
    print(f"\nGenealogy nodes: {len(history.genealogy_tree)}")
    print(f"Best-individual ancestors: {len(tree)}")


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=40)
    history = tools.History()
    history.update(pop)
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)
    hof = tools.HallOfFame(1)
    tools.ea_simple(
        toolbox,
        pop,
        generations=12,
        cx_prob=0.5,
        mut_prob=0.2,
        hof=hof,
        stats=stats,
        verbose=True,
    )
    print_results(history, hof[0])


if __name__ == "__main__":
    main()
