from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)

DIM = 100
GENS = 250
TARGET = 1e-6


def make_toolbox():
    creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
    creator.create_type("Individual", list, fitness=creator.FitnessMin)
    strategy = tools.StrategySeparable(
        centroid=[3.0] * DIM,
        sigma=3.0,
        offsprings=80,
        low=-5.0,
        up=5.0,
    )
    toolbox = Toolbox()
    toolbox.register("evaluate", tools.bm_sphere)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    return toolbox


def main():
    toolbox = make_toolbox()
    hof = tools.HallOfFame(1)
    tools.ea_generate_update(toolbox, generations=GENS, hof=hof, verbose=True)
    best = hof[0].fitness.values[0]
    if best >= TARGET:
        raise RuntimeError("Separable CMA did not reach the sphere target.")
    print(f"Reached sphere fitness {best:.3e} in {DIM} dimensions.")


if __name__ == "__main__":
    main()
