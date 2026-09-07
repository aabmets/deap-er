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
import pytest
from deap_er import Fitness, creator, tools

MIG_FIT = "MIG_FIT"
MIG_IND = "MIG_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(MIG_FIT, Fitness, weights=(1.0,))
    creator.create_type(MIG_IND, list, fitness=creator.__dict__[MIG_FIT])
    yield creator.__dict__[MIG_IND]
    del creator.__dict__[MIG_FIT]
    del creator.__dict__[MIG_IND]


def _demes(ind_cls, nbr_demes=3, size=4):
    demes = []
    for deme in range(nbr_demes):
        members = []
        for member in range(size):
            value = deme * 10 + member
            ind = ind_cls([value])
            ind.fitness.values = (float(value),)
            members.append(ind)
        demes.append(members)
    return demes


def test_mig_ring_without_replacement(ind_cls):
    demes = _demes(ind_cls)

    tools.mig_ring(demes, 2, tools.sel_best)

    # Each deme sends its two best to the next deme, which drops its own two best.
    assert sorted(ind[0] for ind in demes[0]) == [0, 1, 22, 23]
    assert sorted(ind[0] for ind in demes[1]) == [2, 3, 10, 11]
    assert sorted(ind[0] for ind in demes[2]) == [12, 13, 20, 21]


def test_mig_ring_with_replacement(ind_cls):
    demes = _demes(ind_cls)

    tools.mig_ring(demes, 2, tools.sel_best, replacement=tools.sel_worst)

    # Emigrants are still the two best, but the vacancies are the two worst.
    assert sorted(ind[0] for ind in demes[0]) == [2, 3, 22, 23]
    assert sorted(ind[0] for ind in demes[1]) == [2, 3, 12, 13]
    assert sorted(ind[0] for ind in demes[2]) == [12, 13, 22, 23]


def test_mig_ring_replaces_by_identity_not_genotype(ind_cls):
    weak = ind_cls([5])
    weak.fitness.values = (1.0,)
    strong = ind_cls([5])
    strong.fitness.values = (50.0,)
    other = ind_cls([1])
    other.fitness.values = (0.0,)
    extra = ind_cls([2])
    extra.fitness.values = (0.0,)
    dest_a = ind_cls([9])
    dest_a.fitness.values = (0.0,)
    dest_b = ind_cls([8])
    dest_b.fitness.values = (0.0,)
    dest_c = ind_cls([7])
    dest_c.fitness.values = (0.0,)
    dest_d = ind_cls([6])
    dest_d.fitness.values = (0.0,)
    demes = [[weak, strong, other, extra], [dest_a, dest_b, dest_c, dest_d]]

    tools.mig_ring(demes, 1, tools.sel_best)

    assert all(member is not strong for member in demes[0])
    assert any(member is weak for member in demes[0])
    assert any(member is strong for member in demes[1])


def test_mig_ring_preserves_deme_sizes(ind_cls):
    demes = _demes(ind_cls, nbr_demes=4, size=5)

    tools.mig_ring(demes, 2, tools.sel_best, replacement=tools.sel_worst)

    assert [len(deme) for deme in demes] == [5, 5, 5, 5]


def test_mig_ring_sel_random_completes_with_duplicate_draws(ind_cls):
    tools.rng.seed(0)
    demes = _demes(ind_cls, nbr_demes=3, size=3)

    tools.mig_ring(demes, 2, tools.sel_random)

    assert [len(deme) for deme in demes] == [3, 3, 3]


def test_mig_ring_duplicate_slots_use_distinct_vacancies(ind_cls):
    def pick_first_twice(population, count):
        return [population[0]] * count

    demes = _demes(ind_cls, nbr_demes=2, size=3)
    dest_ids = [id(member) for member in demes[1]]

    tools.mig_ring(demes, 2, pick_first_twice)

    changed = sum(1 for i, member in enumerate(demes[1]) if id(member) != dest_ids[i])
    assert changed == 2


def test_mig_ring_overlapping_destinations_completes(ind_cls):
    demes = _demes(ind_cls, nbr_demes=3, size=3)

    tools.mig_ring(demes, 2, tools.sel_best, mig_indices=[1, 1, 0])

    assert [len(deme) for deme in demes] == [3, 3, 3]


def test_mig_ring_unequal_deme_sizes_completes(ind_cls):
    small = _demes(ind_cls, nbr_demes=1, size=2)[0]
    large = []
    for value in (10, 11, 12, 13, 14):
        member = ind_cls([value])
        member.fitness.values = (float(value),)
        large.append(member)
    demes = [small, large]

    tools.mig_ring(demes, 3, tools.sel_best)

    assert [len(deme) for deme in demes] == [2, 5]
    assert {member[0] for member in demes[0]} & {12, 13, 14}


def test_mig_ring_unequal_ring_does_not_alias_across_demes(ind_cls):
    small = _demes(ind_cls, nbr_demes=1, size=1)[0]
    mid = []
    for value in (10, 11, 12, 13, 14):
        member = ind_cls([value])
        member.fitness.values = (float(value),)
        mid.append(member)
    last = []
    for value in (20, 21, 22, 23):
        member = ind_cls([value])
        member.fitness.values = (float(value),)
        last.append(member)
    demes = [small, mid, last]

    tools.mig_ring(demes, 3, tools.sel_best)

    assert [len(deme) for deme in demes] == [1, 5, 4]
    seen: dict[int, int] = {}
    for deme_idx, deme in enumerate(demes):
        for member in deme:
            key = id(member)
            assert key not in seen
            seen[key] = deme_idx

    others_before = [[member[0] for member in deme] for deme in (demes[0], demes[2])]
    demes[1][0][0] = 999
    others_after = [[member[0] for member in deme] for deme in (demes[0], demes[2])]
    assert others_after == others_before


def test_mig_ring_sel_random_oversize_count_completes(ind_cls):
    tools.rng.seed(1)
    demes = _demes(ind_cls, nbr_demes=2, size=2)

    tools.mig_ring(demes, 5, tools.sel_random)

    assert [len(deme) for deme in demes] == [2, 2]


def test_mig_ring_replacement_does_not_alias_across_demes(ind_cls):
    demes = _demes(ind_cls, nbr_demes=2, size=3)

    tools.mig_ring(demes, 1, tools.sel_best, replacement=tools.sel_worst)

    for src in demes[0]:
        for dst in demes[1]:
            assert src is not dst

    source_before = [member[0] for member in demes[0]]
    demes[1][0][0] = 999
    assert [member[0] for member in demes[0]] == source_before
