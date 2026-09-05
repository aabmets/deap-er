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
from deap_er import base, creator, tools

MIG_FIT = "MIG_FIT"
MIG_IND = "MIG_IND"


@pytest.fixture
def ind_cls():
    creator.create(MIG_FIT, base.Fitness, weights=(1.0,))
    creator.create(MIG_IND, list, fitness=creator.__dict__[MIG_FIT])
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
