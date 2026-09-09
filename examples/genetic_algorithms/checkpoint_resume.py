import tempfile
from pathlib import Path

from deap_er import Checkpoint, Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

N_BITS = 24
POP = 60
FIRST_GENS = 8
RESUME_GENS = 40
CHECKPOINT_FILE = "onemax.dcpf"


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
    return toolbox


def step_generation(toolbox, population):
    offspring = tools.var_and(toolbox, population, 0.5, 0.2)
    tools.evaluate_invalid(toolbox, offspring)
    return toolbox.select(offspring, len(offspring))


def run_phase(toolbox, cp, generations):
    if not cp.is_loaded():
        cp.pop = toolbox.population(size=POP)
        tools.evaluate_invalid(toolbox, cp.pop)
        cp.hof = tools.HallOfFame(1)
        cp.hof.update(cp.pop)
    for _gen in cp.range(generations):
        cp.pop[:] = step_generation(toolbox, cp.pop)
        cp.hof.update(cp.pop)


def print_results(best_ind, restored_fit, saved_fit):
    if restored_fit != saved_fit:
        raise RuntimeError("Restored hall-of-fame fitness does not match the saved run.")
    if not all(gene == 1 for gene in best_ind):
        raise RuntimeError("Resume failed to reach all-ones.")
    print("\nCheckpoint restored and resume converged.")


def main():
    toolbox = setup()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        first = Checkpoint(
            CHECKPOINT_FILE,
            dir_path=path,
            raise_errors=True,
            autoload=False,
            hof_ind_cls=creator.Individual,
        )
        first.save_freq = 0
        run_phase(toolbox, first, FIRST_GENS)
        first.generation = FIRST_GENS
        first.save()
        saved_fit = first.hof[0].fitness.values

        second = Checkpoint(
            CHECKPOINT_FILE,
            dir_path=path,
            raise_errors=True,
            hof_ind_cls=creator.Individual,
        )
        if not second.is_loaded():
            raise RuntimeError("Second process failed to load the checkpoint.")
        if getattr(second, "generation", None) != FIRST_GENS:
            raise RuntimeError("Restored generation does not match the saved run.")
        restored_fit = second.hof[0].fitness.values
        run_phase(toolbox, second, RESUME_GENS)
        print_results(second.hof[0], restored_fit, saved_fit)
        path.joinpath(CHECKPOINT_FILE).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
