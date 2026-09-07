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
from deap_er.private.records.logbook import Logbook


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


def test_pop_without_generation_removes_positional_chapter_row():
    logbook = Logbook()
    logbook.record(nevals=10, size={"avg": 1.0})
    logbook.record(nevals=20, size={"avg": 2.0})
    assert logbook.pop(0)["nevals"] == 10
    assert [entry["nevals"] for entry in logbook] == [20]
    assert [entry["avg"] for entry in logbook.chapters["size"]] == [2.0]
    data = [line.split("\t") for line in str(logbook).splitlines() if "20" in line]
    assert data
    assert any(cell.strip() == "2" for cell in data[0])


def test_pop_without_generation_keeps_shorter_chapter():
    logbook = Logbook()
    logbook.record(nevals=10)
    logbook.record(nevals=20, size={"avg": 2.0})
    logbook.pop(0)
    assert [entry["nevals"] for entry in logbook] == [20]
    assert [entry["avg"] for entry in logbook.chapters["size"]] == [2.0]


def test_pop_removes_matching_chapter_row():
    logbook = Logbook()
    logbook.record(gen=0, size={"avg": 10})
    logbook.record(gen=1, size={"avg": 20})
    logbook.record(gen=2, size={"avg": 30})
    assert logbook.pop(1)["gen"] == 1
    assert [entry["gen"] for entry in logbook] == [0, 2]
    assert [entry["avg"] for entry in logbook.chapters["size"]] == [10, 30]


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


def test_delitem_slice_does_not_drop_unrelated_chapter_row():
    logbook = Logbook()
    logbook.record(gen=0)
    logbook.record(gen=1, size={"avg": 1})
    del logbook[0:1]
    assert [entry["gen"] for entry in logbook] == [1]
    assert [entry["avg"] for entry in logbook.chapters["size"]] == [1]

    cleared = Logbook()
    cleared.record(gen=0)
    cleared.record(gen=1, size={"avg": 1})
    del cleared[:]
    assert list(cleared) == []
    assert list(cleared.chapters["size"]) == []


def test_delitem_pairs_matching_repeat_generation():
    logbook = Logbook()
    logbook.record(gen=1, size={"avg": 10})
    logbook.record(gen=1, size={"avg": 20})
    del logbook[1]
    assert [entry["gen"] for entry in logbook] == [1]
    assert [entry["avg"] for entry in logbook.chapters["size"]] == [10]


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


def test_empty_logbook_with_header_prints_banner_once():
    logbook = Logbook()
    logbook.header = ["gen", "nevals"]
    banner = logbook.stream
    assert "gen" in banner
    assert "empty" not in banner.lower()
    logbook.record(gen=0, nevals=3)
    follow = logbook.stream
    assert "gen" not in follow
    assert "3" in follow


def test_empty_stream_without_header_still_prints_banner_on_first_record():
    logbook = Logbook()
    assert logbook.stream == "The Logbook is empty."
    logbook.record(gen=0, nevals=3)
    first = logbook.stream
    assert "gen" in first
    assert "3" in first


def test_stream_omitted_chapter_does_not_raise():
    logbook = Logbook()
    logbook.header = ["gen", "size"]
    logbook.chapters["size"].header = ["avg"]
    logbook.record(gen=0, size={"avg": 10.0})
    first = logbook.stream
    assert "10" in first
    logbook.record(gen=1)
    second = logbook.stream
    assert "1" in second
    assert "10" not in second
    logbook.record(gen=2, size={"avg": 30.0})
    third = logbook.stream
    assert "30" in third


def test_str_keeps_chapter_values_on_their_generation():
    logbook = Logbook()
    logbook.header = ["gen", "size"]
    logbook.chapters["size"].header = ["avg"]
    logbook.record(gen=0, size={"avg": 10.0})
    logbook.record(gen=1)
    logbook.record(gen=2, size={"avg": 30.0})

    lines = str(logbook).splitlines()
    data = [line.split("\t") for line in lines if line[:1].isdigit()]
    assert [row[0].strip() for row in data] == ["0", "1", "2"]
    assert data[0][1].strip() == "10"
    assert data[1][1].strip() == ""
    assert data[2][1].strip() == "30"


def test_json_round_trip_restores_chapters_and_numpy_scalars():
    import numpy

    logbook = Logbook()
    logbook.header = ["gen"]
    logbook.record(gen=0, score=numpy.float64(1.5), size={"avg": 4})
    restored = Logbook.from_json(logbook.to_json())
    assert restored.header == ["gen"]
    assert restored[0]["score"] == 1.5
    assert restored.chapters["size"][0]["avg"] == 4


def test_json_round_trip_restores_nested_chapters():
    logbook = Logbook()
    logbook.record(gen=0, fit={"avg": 1, "more": {"x": 2}})
    restored = Logbook.from_json(logbook.to_json())
    assert list(restored.chapters["fit"].chapters.keys()) == ["more"]
    assert restored.chapters["fit"].chapters["more"][0]["x"] == 2
    assert restored.chapters["fit"][0]["avg"] == 1
