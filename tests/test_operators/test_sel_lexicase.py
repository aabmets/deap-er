#
#   Apache License 2.0
#
#   Copyright (c) 2022, Mattias Aabmets
#
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#
#   SPDX-License-Identifier: Apache-2.0
#
from deap_er import tools


def test_epsilon_lexicase_zero_epsilon_is_strict(single_obj, make):
    # An explicit epsilon of 0.0 must not be treated as "compute it from the MAD".
    best = make(single_obj, [0], (10.0,))
    population = [best] + [make(single_obj, [i], (v,)) for i, v in enumerate((9.0, 8.0, 0.0), 1)]

    chosen = tools.sel_epsilon_lexicase(population, 20, epsilon=0.0)

    assert all(ind is best for ind in chosen)


def test_epsilon_lexicase_recomputes_epsilon_for_each_selection(multi_obj, make):
    # The two cases have very different spreads, so reusing the epsilon of one
    # case while filtering on the other changes which candidates survive.
    values = [(10.0, 0.0), (9.0, 50.0), (8.0, 100.0), (7.0, 150.0)]
    population = [make(multi_obj, [i], value) for i, value in enumerate(values)]

    tools.rng.seed(99)
    batched = tools.sel_epsilon_lexicase(population, 8)
    tools.rng.seed(99)
    one_at_a_time = [tools.sel_epsilon_lexicase(population, 1)[0] for _ in range(8)]

    # Each selection must start from a fresh epsilon, so a batch of eight has to
    # match eight independent single selections drawn from the same seed.
    assert batched == one_at_a_time
