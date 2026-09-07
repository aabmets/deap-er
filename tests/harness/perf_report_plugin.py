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
import sys
import typing as t
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.harness.measure import (
    bind_perf_nodeid,
    take_perf_sections,
    unbind_perf_nodeid,
)

__all__ = [
    "flagged_tests",
    "pytest_addoption",
    "pytest_configure",
    "pytest_runtest_call",
    "pytest_runtest_logreport",
    "pytest_runtest_makereport",
    "pytest_sessionfinish",
    "pytest_terminal_summary",
]

DEFAULT_PERF_REPORT = Path("reports") / "unittest_performance.json"
SLOW_TEST_SECONDS = 0.5
PROP_PARAMS = "perf_params"
PROP_PATH = "perf_path"
PROP_QUALNAME = "perf_qualname"
PROP_SECTIONS = "perf_sections"
PROP_WORKER = "perf_worker"

_controller: pytest.Config | None = None
_records: dict[str, dict[str, t.Any]] = {}
_report_path: Path | None = None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--perf-report",
        default=str(DEFAULT_PERF_REPORT),
        help="Unittest performance JSON path, relative to the pytest rootpath.",
    )


def pytest_configure(config: pytest.Config) -> None:
    global _controller, _records, _report_path
    if hasattr(config, "workerinput"):
        return
    _controller = config
    _records = {}
    _report_path = None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> t.Generator[None, t.Any, t.Any]:
    token = bind_perf_nodeid(item.nodeid)
    try:
        result = yield
        return result
    finally:
        unbind_perf_nodeid(token)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[t.Any],
) -> t.Generator[None, t.Any, pytest.TestReport]:
    report = yield
    assert isinstance(report, pytest.TestReport)
    worker = worker_id(item.config)
    sections = take_perf_sections(item.nodeid) if call.when == "call" else {}
    report.user_properties.extend(
        [
            (PROP_PATH, item_path(item)),
            (PROP_QUALNAME, item_qualname(item)),
            (PROP_PARAMS, item_params(item)),
            (PROP_WORKER, worker),
            (PROP_SECTIONS, sections),
        ]
    )
    return report


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if _controller is None or not should_record(report):
        return
    props = dict(report.user_properties)
    raw = props.get(PROP_SECTIONS)
    sections_raw = raw if isinstance(raw, dict) else {}
    sections = {label: {"duration_s": duration} for label, duration in sections_raw.items()}
    duration = 0.0 if report.when != "call" else float(report.duration)
    _records[report.nodeid] = {
        "path": props.get(PROP_PATH) or report.nodeid.split("::", 1)[0],
        "qualname": props.get(PROP_QUALNAME) or report.nodeid,
        "params": props.get(PROP_PARAMS),
        "outcome": report_outcome(report),
        "duration_s": duration,
        "worker": props.get(PROP_WORKER),
        "sections": sections,
    }


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    global _report_path
    if _controller is None:
        return
    dest = perf_report_path(_controller)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": build_meta(_controller, _records),
        "flagged": flagged_tests(_records),
        "tests": {nodeid: _records[nodeid] for nodeid in sorted(_records)},
    }
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _report_path = dest


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: pytest.ExitCode,
    config: pytest.Config,
) -> None:
    if hasattr(config, "workerinput") or _report_path is None:
        return
    try:
        display = _report_path.relative_to(config.rootpath).as_posix()
    except ValueError:
        display = _report_path.as_posix()
    terminalreporter.write_sep("=", "UNITTEST PERFORMANCE")
    terminalreporter.write_line(f"Performance report written to {display}")


def item_path(item: pytest.Item) -> str:
    path = getattr(item, "path", None)
    if path is None:
        return item.nodeid.split("::", 1)[0]
    try:
        return Path(path).relative_to(item.config.rootpath).as_posix()
    except ValueError:
        return Path(path).as_posix()


def item_qualname(item: pytest.Item) -> str:
    module = getattr(item, "module", None)
    func = getattr(item, "function", None)
    if module is None or func is None:
        return item.nodeid.replace("/", ".").replace("::", ".")
    parts = [str(module.__name__)]
    cls = getattr(item, "cls", None)
    if cls is not None:
        parts.append(str(cls.__name__))
    parts.append(str(func.__name__))
    return ".".join(parts)


def item_params(item: pytest.Item) -> str | None:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return None
    ident = getattr(callspec, "id", None)
    return None if ident is None else str(ident)


def worker_id(config: pytest.Config) -> str | None:
    workerinput = getattr(config, "workerinput", None)
    if not isinstance(workerinput, dict):
        return None
    ident = workerinput.get("workerid")
    return None if ident is None else str(ident)


def should_record(report: pytest.TestReport) -> bool:
    if report.when == "call":
        return True
    return report.when == "setup" and report.outcome in {"skipped", "failed"}


def report_outcome(report: pytest.TestReport) -> str:
    if report.when != "call" and report.failed:
        return "error"
    return report.outcome


def perf_report_path(config: pytest.Config) -> Path:
    raw = Path(str(config.getoption("perf_report")))
    return raw if raw.is_absolute() else config.rootpath / raw


def flagged_tests(records: dict[str, dict[str, t.Any]]) -> list[str]:
    return [
        nodeid
        for nodeid in sorted(records)
        if float(records[nodeid].get("duration_s", 0.0)) >= SLOW_TEST_SECONDS
    ]


def build_meta(config: pytest.Config, records: dict[str, dict[str, t.Any]]) -> dict[str, t.Any]:
    workers = sorted({row["worker"] for row in records.values() if row.get("worker")})
    nopt = config.getoption("numprocesses", default=0)
    if nopt in (0, "0", None):
        return meta_fields(False, 0, [])
    if isinstance(nopt, int):
        return meta_fields(True, nopt, workers)
    return meta_fields(True, len(workers), workers)


def meta_fields(
    xdist: bool,
    numprocesses: int,
    workers: list[str],
) -> dict[str, t.Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "xdist": xdist,
        "numprocesses": numprocesses,
        "workers": workers,
        "python": sys.version.split()[0],
    }
