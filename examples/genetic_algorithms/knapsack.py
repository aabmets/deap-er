from deap_er import algorithms as algos
from deap_er import operators as ops
from deap_er import utilities as utils
from deap_er import records
from deap_er import creator
from deap_er import base
import string
import random


random.seed(1234)  # ensure reproducibility

IND_INIT_SIZE = 5
MAX_ITEM = 50
MAX_WEIGHT = 50
NBR_ITEMS = 20
NAME_LEN = 3

items = dict()


def create_items():
    alphabet = list(string.ascii_uppercase)
    for _ in range(NBR_ITEMS):
        while True:
            name = ''.join(random.choice(alphabet) for _ in range(NAME_LEN))
            if name not in items:
                break
        weight = random.randint(1, 10)
        value = random.uniform(0, 100)
        items.update(
            {name: (weight, value)}
        )


def evaluate(individual: set) -> tuple[int, int]:
    if len(individual) <= MAX_ITEM:
        _weight, _value = 0, 0
        for item in individual:
            _weight += items[item][0]
            _value += items[item][1]
        if _weight <= MAX_WEIGHT:
            return _weight, _value
    return 10000, 0


def mate(ind1: set, ind2: set) -> tuple[set, set]:
    temp = set(ind1)
    ind1 &= ind2
    ind2 ^= temp
    return ind1, ind2


def mutate(individual: set) -> tuple[set]:
    if random.random() < 0.5:
        if len(individual) > 0:
            items_ = sorted(tuple(individual))
            choice = random.choice(items_)
            individual.remove(choice)
    else:
        names = list(items.keys())
        individual.add(random.choice(names))
    return individual,  # The comma is essential here.


def setup():
    creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
    creator.create("Individual", set, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("attr_item", random.choice, list(items.keys()))
    toolbox.register("individual", utils.init_repeat, creator.Individual, toolbox.attr_item, IND_INIT_SIZE)
    toolbox.register("population", utils.init_repeat, list, toolbox.individual)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)
    toolbox.register("select", ops.sel_nsga_2)
    toolbox.register("evaluate", evaluate)

    return toolbox


def print_results(hof):
    best_ind = sorted(list(hof[-1]))
    best_weight, best_value = 0, 0
    for idx in best_ind:
        best_weight += items[idx][0]
        best_value += round(items[idx][1], 2)

    keys = sorted(list(items.keys()))
    for key in tuple(keys):
        if key not in best_ind:
            keys.remove(key)
            keys.append(key)

    print('\nAvailable items to choose from:')
    print('Names:\t\t' + '\t\t'.join([str(k) for k in keys]))
    print('Weights:\t' + '\t\t'.join([str(items[k][0]) for k in keys]))
    print('Values:\t\t' + '\t'.join([str(round(items[k][1], 2)) for k in keys]))
    print(f'\nItems chosen: {best_ind}')
    print(f'Total weight of chosen items: {best_weight}')
    print(f'Total value of chosen items: {best_value}.')


def main():
    create_items()
    toolbox = setup()
    pop = toolbox.population(size=100)
    hof = records.ParetoFront()

    args = dict(
        toolbox=toolbox,
        population=pop,
        generations=110,
        offsprings=100,
        survivors=50,
        cx_prob=0.5,
        mut_prob=0.2,
        hof=hof
    )
    algos.ea_mu_plus_lambda(**args)
    print_results(hof)


if __name__ == "__main__":
    main()                 
