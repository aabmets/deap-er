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
import json

import numpy
import pytest
from deap_er.private.records.logbook import Logbook


def test_select_multiple_names_and_pop_streamed_row():
    logbook = Logbook()
    logbook.record(gen=0, fit=1.0)
    logbook.record(gen=1, fit=2.0)
    _ = logbook.stream

    gens, fits = logbook.select("gen", "fit")
    assert gens == [0, 1]
    assert fits == [1.0, 2.0]

    removed = logbook.pop(0)
    assert removed["gen"] == 0
    assert logbook.select("gen") == [1]


def test_delete_row_without_generation_and_out_of_range_index():
    logbook = Logbook()
    logbook.record(fit=1.0)
    logbook.record(gen=1, size={"avg": 4})
    del logbook[0]
    assert [entry.get("gen") for entry in logbook] == [1]


def test_json_ready_nested_arrays_and_unknown_types():
    logbook = Logbook()
    logbook.record(gen=0, arr=numpy.array([1, 2]), vals=(3, 4), extra=object())

    payload = json.loads(logbook.to_json())
    entry = payload["entries"][0]
    assert entry["arr"] == [1, 2]
    assert entry["vals"] == [3, 4]
    assert isinstance(entry["extra"], str)


def test_clear_removes_chapter_rows():
    logbook = Logbook()
    logbook.record(gen=0, size={"avg": 10})
    logbook.record(gen=1, size={"avg": 20})
    logbook.clear()
    assert list(logbook) == []
    assert list(logbook.chapters["size"]) == []
    logbook.record(gen=0, size={"avg": 99})
    assert [entry["avg"] for entry in logbook.chapters["size"]] == [99]


def test_clear_after_stream_does_not_drop_new_rows():
    logbook = Logbook()
    logbook.record(gen=0, nevals=1)
    _ = logbook.stream
    logbook.clear()
    logbook.record(gen=1, nevals=9)
    text = logbook.stream
    assert "1" in text
    assert "9" in text


def test_pop_out_of_range_keeps_stream_cursor():
    logbook = Logbook()
    logbook.record(gen=0)
    logbook.record(gen=1)
    _ = logbook.stream
    with pytest.raises(IndexError):
        logbook.pop(-10)
    assert logbook.buff_index == 2
    logbook.record(gen=2)
    assert logbook.stream.split() == ["2"]


def test_chapter_without_first_generation_row_renders_its_columns():
    logbook = Logbook()
    logbook.record(gen=0, x=1)
    logbook.record(gen=1, x=2, fit={"avg": 3})
    lines = str(logbook).splitlines()
    assert lines[2].split() == ["gen", "x", "avg", "gen", "x"]
    assert lines[-1].split() == ["1", "2", "3", "1", "2"]


def test_streamed_chapter_keeps_column_widths_across_reads():
    logbook = Logbook()
    logbook.header = ["gen", "fit"]
    logbook.chapters["fit"].header = ["avg", "max"]
    logbook.record(gen=0, fit={"avg": 123456.0, "max": 98765.5})
    first = logbook.stream.splitlines()
    logbook.record(gen=1, fit={"avg": 1.5, "max": 2.5})
    second = logbook.stream.splitlines()
    assert len(second) == 1
    assert len(second[0]) == len(first[-1])
    assert second[0] == str(logbook).splitlines()[-1]


def test_stream_header_after_enabling_log_header_includes_chapter_columns():
    logbook = Logbook()
    logbook.log_header = False
    logbook.record(gen=0, x=1, fit={"avg": 1})
    _ = logbook.stream
    logbook.log_header = True
    logbook.record(gen=1, x=2, fit={"avg": 2})
    lines = logbook.stream.splitlines()
    assert lines[0].split() == ["fit"]
    assert set(lines[1].strip()) == {"-"}
    assert lines[2].split() == ["gen", "x", "avg", "gen", "x"]
    assert lines[3].split() == ["1", "2", "2", "1", "2"]


def test_remove_drops_chapter_row_and_keeps_stream_cursor():
    logbook = Logbook()
    logbook.record(gen=0, fit={"avg": 1})
    logbook.record(gen=1, fit={"avg": 2})
    _ = logbook.stream
    logbook.remove(logbook[0])
    assert logbook.chapters["fit"].select("gen") == [1]
    logbook.record(gen=2, fit={"avg": 3})
    assert logbook.stream.split() == ["2", "3", "2"]
