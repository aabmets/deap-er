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
from deap_er.records.logbook import Logbook


def _filled(count=5):
    logbook = Logbook()
    for gen in range(count):
        logbook.record(gen=gen)
    return logbook


def test_delitem_slice_removes_exactly_the_selected_rows():
    logbook = _filled()

    del logbook[0:2]

    assert [entry["gen"] for entry in logbook] == [2, 3, 4]


def test_delitem_slice_with_step():
    logbook = _filled()

    del logbook[0:5:2]

    assert [entry["gen"] for entry in logbook] == [1, 3]


def test_delitem_slice_keeps_chapters_aligned():
    logbook = Logbook()
    for gen in range(4):
        logbook.record(gen=gen, size={"avg": gen})

    del logbook[0:2]

    assert [entry["gen"] for entry in logbook] == [2, 3]
    assert [entry["avg"] for entry in logbook.chapters["size"]] == [2, 3]


def test_delitem_single_index():
    logbook = _filled()

    del logbook[1]

    assert [entry["gen"] for entry in logbook] == [0, 2, 3, 4]
