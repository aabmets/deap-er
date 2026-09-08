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
from deap_er import records


def test_regression_generalization_gap_chapter_contents():
    logbook = records.Logbook()
    records.record_policy_generalization_gap(
        logbook,
        gen=3,
        train_score=8.0,
        held_out_score=3.0,
        nevals=20,
    )
    chapter = logbook.chapters[records.POLICY_GENERALIZATION_GAP_CHAPTER][0]
    assert chapter == {
        "gen": 3,
        "nevals": 20,
        "train": 8.0,
        "held_out": 3.0,
        "gap": 5.0,
    }


def test_policy_generalization_gap_chapter_fields():
    chapter = records.policy_generalization_gap(5.0, 2.0)
    assert chapter == {"train": 5.0, "held_out": 2.0, "gap": 3.0}


def test_policy_generalization_gap_without_held_out():
    chapter = records.policy_generalization_gap(4.0, None)
    assert chapter == {"train": 4.0, "held_out": None, "gap": None}


def test_record_policy_generalization_gap_logbook_chapter():
    logbook = records.Logbook()
    records.record_policy_generalization_gap(
        logbook,
        gen=0,
        train_score=6.0,
        held_out_score=1.0,
        nevals=12,
    )
    assert logbook[0]["gen"] == 0
    assert logbook[0]["nevals"] == 12
    chapter = logbook.chapters[records.POLICY_GENERALIZATION_GAP_CHAPTER][0]
    assert chapter["gen"] == 0
    assert chapter["nevals"] == 12
    assert chapter["train"] == 6.0
    assert chapter["held_out"] == 1.0
    assert chapter["gap"] == 5.0


def test_generalization_gap_chapter_constant():
    assert records.POLICY_GENERALIZATION_GAP_CHAPTER == "generalization_gap"
