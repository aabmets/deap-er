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
import math
import pickle

import dill
import pytest
from deap_er import Checkpoint, Fitness, creator, tools

KEY_FIT = "KEY_MO_FIT"
KEY_IND = "KEY_MO_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(KEY_FIT, Fitness, weights=(1.0, -1.0))
    creator.create_type(KEY_IND, list, fitness=creator.__dict__[KEY_FIT])
    yield creator.__dict__[KEY_IND]
    del creator.__dict__[KEY_FIT]
    del creator.__dict__[KEY_IND]


def _ind(ind_cls, gene, values):
    individual = ind_cls([gene])
    individual.fitness.values = values
    return individual


def mean_value(individual):
    return sum(individual.fitness.values) / len(individual.fitness.values)


def test_hof_ranks_multi_objective_individuals_by_key(ind_cls):
    hof = tools.HallOfFame(2, key=mean_value)
    hof.update(
        [
            _ind(ind_cls, 0, (5.0, 1.0)),
            _ind(ind_cls, 1, (1.0, 9.0)),
            _ind(ind_cls, 2, (2.0, 0.0)),
            _ind(ind_cls, 3, (0.0, 0.0)),
        ]
    )
    assert [ind[0] for ind in hof] == [1, 0]
    assert hof.keys == [3.0, 5.0]


def test_hof_key_direction_is_larger_is_better(ind_cls):
    hof = tools.HallOfFame(1, key=lambda ind: -mean_value(ind))
    hof.update([_ind(ind_cls, 0, (5.0, 1.0)), _ind(ind_cls, 1, (0.0, 0.0))])
    assert hof[0][0] == 1


def test_hof_similar_member_is_replaced_only_when_key_improves(ind_cls):
    hof = tools.HallOfFame(3, similar=lambda a, b: a[0] == b[0], key=mean_value)
    hof.update([_ind(ind_cls, 0, (2.0, 2.0))])
    hof.update([_ind(ind_cls, 0, (1.0, 1.0))])
    assert hof[0].fitness.values == (2.0, 2.0)
    hof.update([_ind(ind_cls, 0, (4.0, 4.0))])
    assert len(hof) == 1
    assert hof[0].fitness.values == (4.0, 4.0)


def test_hof_skips_non_finite_key(ind_cls):
    hof = tools.HallOfFame(3, key=lambda ind: math.nan if ind[0] else 1.0)
    hof.update([_ind(ind_cls, 0, (1.0, 1.0)), _ind(ind_cls, 1, (2.0, 2.0))])
    assert [ind[0] for ind in hof] == [0]


def test_hof_from_json_restores_key_order(ind_cls):
    hof = tools.HallOfFame(3, key=mean_value)
    hof.update([_ind(ind_cls, 0, (5.0, 1.0)), _ind(ind_cls, 1, (1.0, 9.0))])
    restored = tools.HallOfFame.from_json(hof.to_json(), ind_cls, key=mean_value)
    assert [ind[0] for ind in restored] == [1, 0]
    assert restored.key is mean_value


def test_hof_with_key_pickles(ind_cls):
    hof = tools.HallOfFame(2, key=mean_value)
    hof.update([_ind(ind_cls, 0, (5.0, 1.0))])
    assert pickle.loads(pickle.dumps(hof)).keys == [3.0]
    lam = tools.HallOfFame(2, key=lambda ind: ind.fitness.values[0])
    lam.update([_ind(ind_cls, 0, (5.0, 1.0))])
    clone = dill.loads(dill.dumps(lam))
    clone.update([_ind(ind_cls, 1, (7.0, 1.0))])
    assert [ind[0] for ind in clone] == [1, 0]


def test_grid_archive_accepts_multi_objective_with_key(ind_cls):
    with pytest.raises(ValueError, match="single-objective"):
        tools.GridArchive(ranges=[(0.0, 1.0)], bins=2).add(_ind(ind_cls, 0, (1.0, 1.0)), (0.1,))
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=2, key=mean_value)
    assert archive.add(_ind(ind_cls, 0, (1.0, 1.0)), (0.1,)) is True
    assert archive.add(_ind(ind_cls, 1, (3.0, -1.0)), (0.1,)) is False
    assert archive.add(_ind(ind_cls, 2, (0.0, 4.0)), (0.1,)) is True
    elite = archive.elite_at((0.1,))
    assert elite is not None and elite[0] == 2
    assert archive.add(_ind(ind_cls, 3, (6.0, 0.0)), (0.9,)) is True
    assert archive.stats.qd_score == pytest.approx(5.0)


def test_grid_archive_negated_key_minimises(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=1, key=lambda ind: -mean_value(ind))
    archive.add(_ind(ind_cls, 0, (4.0, 4.0)), (0.5,))
    assert archive.add(_ind(ind_cls, 1, (1.0, 1.0)), (0.5,)) is True
    assert archive.add(_ind(ind_cls, 2, (2.0, 2.0)), (0.5,)) is False


def test_grid_archive_rejects_non_finite_key(ind_cls):
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=1, key=lambda ind: math.inf)
    assert archive.add(_ind(ind_cls, 0, (1.0, 1.0)), (0.5,)) is False
    assert len(archive) == 0


def test_records_with_lambda_key_survive_checkpoint(ind_cls, tmp_path):
    hof = tools.HallOfFame(2, key=lambda ind: ind.fitness.values[1])
    archive = tools.GridArchive(ranges=[(0.0, 1.0)], bins=2, key=lambda ind: ind.fitness.values[1])
    for gene, values in enumerate([(0.0, 1.0), (0.0, 3.0)]):
        hof.update([_ind(ind_cls, gene, values)])
        archive.add(_ind(ind_cls, gene, values), (0.1,))
    cpt = Checkpoint(file_name="key.dcpf", dir_path=tmp_path, raise_errors=True)
    cpt.hof = hof
    cpt.archive = archive
    assert cpt.save() is True
    loaded = Checkpoint(file_name="key.dcpf", dir_path=tmp_path, raise_errors=True)
    assert [ind[0] for ind in loaded.hof] == [1, 0]
    assert loaded.archive.elite_at((0.1,))[0] == 1
    assert loaded.archive.add(_ind(ind_cls, 2, (0.0, 9.0)), (0.1,)) is True
