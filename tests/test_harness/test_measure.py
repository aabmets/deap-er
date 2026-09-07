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
from tests.harness import measure_section
from tests.harness.measure import bind_perf_nodeid, take_perf_sections, unbind_perf_nodeid


def test_measure_section_requires_running_test():
    token = bind_perf_nodeid(None)
    try:
        with (
            pytest.raises(RuntimeError, match="inside a running pytest test"),
            measure_section("x"),
        ):
            pass
    finally:
        unbind_perf_nodeid(token)


def test_measure_section_rejects_empty_label():
    token = bind_perf_nodeid("fake::test")
    try:
        with pytest.raises(ValueError, match="non-empty"), measure_section(""):
            pass
        with pytest.raises(ValueError, match="non-empty"), measure_section("   "):
            pass
    finally:
        unbind_perf_nodeid(token)
        take_perf_sections("fake::test")


def test_measure_section_rejects_duplicate_label():
    token = bind_perf_nodeid("fake::dup")
    try:
        with measure_section("once"):
            pass
        with pytest.raises(ValueError, match="already used"), measure_section("once"):
            pass
    finally:
        unbind_perf_nodeid(token)
        take_perf_sections("fake::dup")


def test_measure_section_records_unique_labels():
    token = bind_perf_nodeid("fake::ok")
    try:
        with measure_section("setup"):
            pass
        with measure_section("evaluate"):
            pass
        sections = take_perf_sections("fake::ok")
        assert set(sections) == {"setup", "evaluate"}
        assert sections["setup"] >= 0.0
        assert sections["evaluate"] >= 0.0
    finally:
        unbind_perf_nodeid(token)
        take_perf_sections("fake::ok")
