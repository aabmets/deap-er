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


def test_pop_negative_index_does_not_rewind_unstreamed_cursor():
    logbook = _filled()
    logbook.buff_index = 2
    logbook.pop(-1)
    assert logbook.buff_index == 2
    streamed_gens = [int(token) for token in logbook.stream.split() if token.isdigit()]
    assert 1 not in streamed_gens


def test_delitem_does_not_drop_unrelated_chapter_row():
    logbook = Logbook()
    logbook.record(gen=0)
    logbook.record(gen=1, size={"avg": 1})
    del logbook[0]
    assert [entry["gen"] for entry in logbook] == [1]
    assert [entry["avg"] for entry in logbook.chapters["size"]] == [1]


def test_str_of_empty_logbook():
    assert str(Logbook()) == "The Logbook is empty."


def test_str_without_header_sorts_columns():
    # Characterization of __txt__: columns fall back to sorted keys, and every
    # column is padded to the widest cell it holds.
    logbook = Logbook()
    for gen in range(3):
        logbook.record(gen=gen, nevals=10 * gen, avg=1.5 * gen)

    assert str(logbook) == (
        "avg\tgen\tnevals\n0  \t0  \t0     \n1.5\t1  \t10    \n3  \t2  \t20    "
    )


def test_str_with_chapters_builds_a_banner_header():
    # Characterization of __txt__: chapter names are centred over their columns
    # above a dashed rule, and the leading gen column is blank on those rows.
    logbook = Logbook()
    logbook.header = ["gen", "fitness", "size"]
    logbook.chapters["fitness"].header = ["min", "max"]
    logbook.chapters["size"].header = ["avg"]
    for gen in range(3):
        logbook.record(
            gen=gen,
            fitness={"min": gen * 1.0, "max": gen * 2.0},
            size={"avg": gen + 0.5},
        )

    assert str(logbook) == (
        "   \t  fitness  \tsize\n"
        "   \t-----------\t--- \n"
        "gen\tmin\tmax\tavg \n"
        "0  \t0  \t0  \t0.5 \n"
        "1  \t1  \t2  \t1.5 \n"
        "2  \t2  \t4  \t2.5 "
    )
