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
