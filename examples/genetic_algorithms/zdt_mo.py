import logging

import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

# Switch this to try each selector: "sms_emoa", "moead", or "age_moea_2".
SELECTOR = "sms_emoa"
DIMENSIONS = 5
BOUND_LOW, BOUND_UP = 0.0, 1.0
SURVIVORS = 24


def make_selector():
    if SELECTOR == "sms_emoa":
        return tools.sel_sms_emoa
    if SELECTOR == "moead":
        weights = tools.uniform_reference_points(2, ref_ppo=12)
        return tools.SelMOEADWithMemory(weights)
    if SELECTOR == "age_moea_2":
        return tools.SelAGE2WithMemory()
    raise ValueError(f"unknown SELECTOR {SELECTOR!r}")


def setup():
    creator.create_type("FitnessMin", Fitness, weights=(-1.0, -1.0))
    creator.create_type("Individual", list, fitness=creator.FitnessMin)

    toolbox = Toolbox()
    toolbox.register("attr_float", tools.rng.uniform, BOUND_LOW, BOUND_UP)
    toolbox.register(
        "individual",
        tools.init_repeat,
        creator.Individual,
        toolbox.attr_float,
        DIMENSIONS,
    )
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register(
        "mate",
        tools.cx_blend_bounded,
        alpha=0.5,
        low=BOUND_LOW,
        up=BOUND_UP,
    )
    toolbox.register(
        "mutate",
        tools.mut_polynomial_bounded,
        low=BOUND_LOW,
        up=BOUND_UP,
        eta=20.0,
        mut_prob=1.0 / DIMENSIONS,
    )
    toolbox.register("evaluate", tools.bm_zdt_1)
    toolbox.register("select", make_selector())

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    return toolbox, stats


def print_results(population, logbook, fronts):
    tools.assign_crowding_dist(population, use_weights=True)
    diverse = tools.sel_tournament_dcd(population, min(8, len(population)))
    hv = tools.hypervolume(population, [11.0, 11.0])
    last_front = fronts[-1]
    print(f"\nSelector: {SELECTOR}")
    print(f"Hypervolume: {hv:.3f}")
    print(f"Last generation front size: {len(last_front)}")
    print("Crowding-diverse subset:")
    for ind in diverse:
        print(f"  {tuple(round(v, 4) for v in ind.fitness.values)}")
    restored = tools.Logbook.from_json(logbook.to_json())
    print(f"Logbook JSON round-trip generations: {restored.select('gen')[-1]}")
    empty = tools.Logbook()
    empty.header = ["gen", "nevals", "duration"]
    print("Empty logbook still prints its header:")
    print(empty)


def main():
    toolbox, stats = setup()
    logger = logging.getLogger("zdt_mo")
    logging.basicConfig(level=logging.INFO)

    pop = toolbox.population(size=SURVIVORS)
    fronts = []
    hof = tools.ParetoFront()
    pop, logbook = tools.ea_mu_plus_lambda(
        toolbox,
        pop,
        generations=20,
        offsprings=SURVIVORS,
        survivors=SURVIVORS,
        cx_prob=0.5,
        mut_prob=0.5,
        hof=hof,
        stats=stats,
        verbose=True,
        logger=logger,
        log_time=True,
        fronts=fronts,
    )
    print_results(pop, logbook, fronts)


if __name__ == "__main__":
    main()
