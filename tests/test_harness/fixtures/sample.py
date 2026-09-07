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
import time

import pytest
from tests.harness import measure_section


def test_plain():
    time.sleep(0.01)


def test_with_sections():
    with measure_section("setup"):
        time.sleep(0.01)
    with measure_section("evaluate"):
        time.sleep(0.02)


def test_nested_sections():
    with measure_section("outer"):
        time.sleep(0.02)
        with measure_section("inner"):
            time.sleep(0.01)


@pytest.mark.skip(reason="perf fixture skip")
def test_skipped():
    pass


@pytest.mark.parametrize("value", ["a", "b"])
def test_param(value):
    assert value in {"a", "b"}


class TestGrouped:
    def test_method(self):
        with measure_section("body"):
            time.sleep(0.01)


def test_duplicate_section():
    with measure_section("once"):
        pass
    with measure_section("once"):
        pass
