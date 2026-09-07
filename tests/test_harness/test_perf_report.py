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
import subprocess
import sys
import typing as t
from pathlib import Path

from tests.harness.perf_report_plugin import flagged_tests

REPO = Path(__file__).resolve().parents[2]
SAMPLE = Path(__file__).resolve().parent / "fixtures" / "sample.py"


def run_sample(tmp_path: Path, workers: str) -> dict[str, t.Any]:
    report = tmp_path / "unittest_performance.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(SAMPLE),
            f"-n{workers}",
            "--noconftest",
            "-p",
            "no:allure_pytest",
            "-p",
            "no:cov",
            "-p",
            "no:timeout",
            "-p",
            "tests.harness.perf_report_plugin",
            "-o",
            "addopts=",
            "-o",
            "testpaths=",
            f"--perf-report={report}",
            "-q",
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert report.is_file(), f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    return json.loads(report.read_text(encoding="utf-8"))


def row_ending(tests: dict[str, t.Any], suffix: str) -> dict[str, t.Any]:
    matches = [row for nodeid, row in tests.items() if nodeid.endswith(suffix)]
    assert len(matches) == 1, suffix
    return matches[0]


def assert_common_report(payload: dict[str, t.Any]) -> None:
    tests = payload["tests"]
    meta = payload["meta"]
    assert "generated_at" in meta
    assert meta["python"]
    assert payload["flagged"] == []
    assert any(key.endswith("::test_plain") for key in tests)

    plain = row_ending(tests, "::test_plain")
    assert plain["outcome"] == "passed"
    assert plain["duration_s"] > 0
    assert plain["path"].endswith("tests/test_harness/fixtures/sample.py")
    assert plain["qualname"].endswith("test_plain")
    assert plain["params"] is None
    assert plain["sections"] == {}

    sections = row_ending(tests, "::test_with_sections")
    assert sections["outcome"] == "passed"
    assert set(sections["sections"]) == {"setup", "evaluate"}
    assert sections["sections"]["setup"]["duration_s"] > 0
    assert sections["sections"]["evaluate"]["duration_s"] > 0
    assert sections["duration_s"] >= sections["sections"]["evaluate"]["duration_s"]

    nested = row_ending(tests, "::test_nested_sections")
    assert set(nested["sections"]) == {"outer", "inner"}
    assert nested["sections"]["outer"]["duration_s"] > nested["sections"]["inner"]["duration_s"]

    skipped = row_ending(tests, "::test_skipped")
    assert skipped["outcome"] == "skipped"
    assert skipped["duration_s"] == 0.0

    param_a = row_ending(tests, "::test_param[a]")
    param_b = row_ending(tests, "::test_param[b]")
    assert param_a["params"] == "a"
    assert param_b["params"] == "b"
    assert param_a["qualname"] == param_b["qualname"]
    assert param_a["qualname"].endswith("test_param")

    grouped = row_ending(tests, "::TestGrouped::test_method")
    assert grouped["qualname"].endswith("TestGrouped.test_method")
    assert grouped["sections"]["body"]["duration_s"] > 0

    duplicate = row_ending(tests, "::test_duplicate_section")
    assert duplicate["outcome"] == "failed"


def test_flagged_tests_lists_nodeids_at_least_500ms():
    records = {
        "tests/fast.py::test_under": {"duration_s": 0.499},
        "tests/slow.py::test_exact": {"duration_s": 0.5},
        "tests/slow.py::test_over": {"duration_s": 0.5001},
        "tests/other.py::test_missing": {},
        "tests/slow.py::test_two": {"duration_s": 2.5},
    }
    assert flagged_tests(records) == [
        "tests/slow.py::test_exact",
        "tests/slow.py::test_over",
        "tests/slow.py::test_two",
    ]


def test_serial_perf_report(tmp_path: Path):
    payload = run_sample(tmp_path, "0")
    assert_common_report(payload)
    assert payload["meta"]["xdist"] is False
    assert payload["meta"]["numprocesses"] == 0
    assert payload["meta"]["workers"] == []
    for row in payload["tests"].values():
        assert row["worker"] is None


def test_xdist_perf_report(tmp_path: Path):
    payload = run_sample(tmp_path, "2")
    assert_common_report(payload)
    assert payload["meta"]["xdist"] is True
    assert payload["meta"]["numprocesses"] == 2
    workers = {row["worker"] for row in payload["tests"].values()}
    assert None not in workers
    assert workers <= {"gw0", "gw1"}
    assert set(payload["meta"]["workers"]) == workers
