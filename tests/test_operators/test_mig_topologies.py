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


MIG_FC_FIT = "MIG_FC_FIT"
MIG_FC_IND = "MIG_FC_IND"
ISL_EVAL_FIT = "ISL_EVAL_FIT"
ISL_EVAL_IND = "ISL_EVAL_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(MIG_FC_FIT, Fitness, weights=(1.0,))
    creator.create_type(MIG_FC_IND, list, fitness=creator.__dict__[MIG_FC_FIT])
    yield creator.__dict__[MIG_FC_IND]
    del creator.__dict__[MIG_FC_FIT]
    del creator.__dict__[MIG_FC_IND]


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


def test_mig_fully_connected_moves_between_all_pairs(ind_cls):
    demes = _demes(ind_cls, nbr_demes=3, size=4)

    tools.mig_fully_connected(demes, 1, tools.sel_best)

    assert [len(deme) for deme in demes] == [4, 4, 4]
    for deme in demes:
        values = {member[0] for member in deme}
        assert len(values) == 4
        assert values & {20, 21, 22, 23}


def test_mig_fully_connected_preserves_deme_sizes(ind_cls):
    demes = _demes(ind_cls, nbr_demes=4, size=5)

    tools.mig_fully_connected(demes, 2, tools.sel_best, replacement=tools.sel_worst)

    assert [len(deme) for deme in demes] == [5, 5, 5, 5]


def test_mig_fully_connected_does_not_alias_across_demes(ind_cls):
    demes = _demes(ind_cls, nbr_demes=2, size=3)

    tools.mig_fully_connected(demes, 1, tools.sel_best, replacement=tools.sel_worst)

    for src in demes[0]:
        for dst in demes[1]:
            assert src is not dst


def test_mig_random_completes_with_seeded_destinations(ind_cls):
    tools.rng.seed(0)
    demes = _demes(ind_cls, nbr_demes=3, size=3)

    tools.mig_random(demes, 2, tools.sel_random)

    assert [len(deme) for deme in demes] == [3, 3, 3]


def test_mig_random_single_deme_is_noop(ind_cls):
    demes = _demes(ind_cls, nbr_demes=1, size=3)
    before = [[member[0] for member in deme] for deme in demes]

    tools.mig_random(demes, 2, tools.sel_best)

    assert [[member[0] for member in deme] for deme in demes] == before


def test_mig_random_with_replacement(ind_cls):
    tools.rng.seed(1)
    demes = _demes(ind_cls, nbr_demes=2, size=3)

    tools.mig_random(demes, 1, tools.sel_best, replacement=tools.sel_worst)

    assert [len(deme) for deme in demes] == [3, 3]
