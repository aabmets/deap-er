from deap_er import Fitness, Toolbox, creator, gp, tools

tools.rng.seed(1234)  # disables randomization

N_BITS = 24
N_DEMES = 3
DEME_SIZE = 30
GENERATIONS = 20
MIG_INTERVAL = 4


def evaluate(individual):
    return (sum(individual),)


def setup(label):
    fit_name = f"Island{label}Fit"
    ind_name = f"Island{label}Ind"
    creator.create_type(fit_name, Fitness, weights=(1.0,))
    creator.create_type(ind_name, list, fitness=getattr(creator, fit_name))
    ind_cls = getattr(creator, ind_name)

    toolbox = Toolbox()
    toolbox.register("attr_bool", tools.rng.randint, 0, 1)
    toolbox.register("individual", tools.init_repeat, ind_cls, toolbox.attr_bool, N_BITS)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("mate", tools.cx_two_point)
    toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.05)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("evaluate", evaluate)
    toolbox.register("clone", tools.clone_individual)

    def vary(population):
        return tools.var_and(toolbox, population, 0.5, 0.2)

    toolbox.register("vary", vary)
    return toolbox


def migrate_ring(populations):
    tools.mig_ring(populations, mig_count=3, selection=tools.sel_best)


def migrate_mesh(populations):
    tools.mig_fully_connected(populations, mig_count=2, selection=tools.sel_best)


def migrate_random(populations):
    tools.mig_random(populations, mig_count=2, selection=tools.sel_best)


def run_topology(label, migrate):
    toolbox = setup(label.replace(" ", ""))
    demes = [(toolbox, toolbox.population(size=DEME_SIZE)) for _ in range(N_DEMES)]
    for gen in range(GENERATIONS):
        step = migrate if gen % MIG_INTERVAL == 0 else None
        tools.step_islands(demes, migrate=step)
    best = max((ind for _, pop in demes for ind in pop), key=lambda ind: ind.fitness.values)
    score = best.fitness.values[0]
    print(f"{label}: best ones {score} / {N_BITS}")
    return score


def main():
    ring_score = run_topology("Ring", migrate_ring)
    mesh_score = run_topology("Fully connected", migrate_mesh)
    random_score = run_topology("Random destinations", migrate_random)
    if max(ring_score, mesh_score, random_score) < N_BITS - 2:
        raise RuntimeError("Island topologies failed to reach the seed-fixed floor.")


if __name__ == "__main__":
    main()
