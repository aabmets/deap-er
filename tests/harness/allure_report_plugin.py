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
import shutil
import subprocess
import sys
import time
import typing as t
import uuid
from pathlib import Path

import pytest

__all__ = ["pytest_sessionstart", "pytest_terminal_summary"]

REPORTS_DIR = Path("reports")
ALLURE_RESULTS_DIR = REPORTS_DIR / "allure-results"
ALLURE_REPORT_DIR = REPORTS_DIR / "allure-report"
COVERAGE_JSON = REPORTS_DIR / "coverage.json"
COVERAGE_MARKDOWN = REPORTS_DIR / "coverage.md"
HTML_COVERAGE_INDEX = Path(".htmlcov/index.html")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _has_allure_results(results_dir: Path) -> bool:
    return any(results_dir.glob("*-result.json"))


def _coverage_percentage(coverage_json: Path) -> str:
    if not coverage_json.exists():
        return "unavailable"

    data = json.loads(coverage_json.read_text(encoding="utf-8"))
    totals = data.get("totals", {})
    display = totals.get("percent_covered_display")
    if isinstance(display, str) and display:
        return display if display.endswith("%") else f"{display}%"

    covered = totals.get("percent_covered")
    if isinstance(covered, int | float):
        return f"{covered:.2f}%"
    return "unavailable"


def _write_environment_properties(results_dir: Path, coverage_total: str) -> None:
    results_dir.joinpath("environment.properties").write_text(
        "\n".join(
            [
                "project = deap-er",
                f"python_version = {sys.version.split()[0]}",
                f"coverage_total = {coverage_total}",
                f"coverage_html = {HTML_COVERAGE_INDEX.as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _copy_attachment(
    source: Path,
    results_dir: Path,
    attachment_uuid: uuid.UUID,
    suffix: str,
) -> str | None:
    if not source.exists():
        return None
    attachment_name = f"{attachment_uuid}-attachment{suffix}"
    shutil.copyfile(source, results_dir / attachment_name)
    return attachment_name


def _write_coverage_result(root: Path) -> None:
    results_dir = root / ALLURE_RESULTS_DIR
    coverage_json = root / COVERAGE_JSON
    coverage_markdown = root / COVERAGE_MARKDOWN
    if not coverage_json.exists() and not coverage_markdown.exists():
        return

    coverage_total = _coverage_percentage(coverage_json)
    _write_environment_properties(results_dir, coverage_total)

    result_uuid = uuid.uuid4()
    markdown_attachment = _copy_attachment(
        coverage_markdown,
        results_dir,
        uuid.uuid4(),
        ".md",
    )
    json_attachment = _copy_attachment(
        coverage_json,
        results_dir,
        uuid.uuid4(),
        ".json",
    )
    attachments: list[dict[str, str]] = []
    if markdown_attachment:
        attachments.append(
            {
                "name": "Coverage Markdown",
                "source": markdown_attachment,
                "type": "text/markdown",
            }
        )
    if json_attachment:
        attachments.append(
            {
                "name": "Coverage JSON",
                "source": json_attachment,
                "type": "application/json",
            }
        )

    timestamp_ms = int(time.time() * 1000)
    result: dict[str, t.Any] = {
        "uuid": str(result_uuid),
        "historyId": "deap-er-coverage-summary",
        "testCaseId": "deap-er-coverage-summary",
        "fullName": "deap_er.coverage#summary",
        "name": "Coverage Summary",
        "status": "passed",
        "stage": "finished",
        "description": (
            f"Total coverage: {coverage_total}. "
            f"Full HTML coverage remains at {HTML_COVERAGE_INDEX.as_posix()}."
        ),
        "attachments": attachments,
        "labels": [
            {"name": "suite", "value": "Coverage"},
            {"name": "framework", "value": "pytest-cov"},
            {"name": "language", "value": "python"},
        ],
        "start": timestamp_ms,
        "stop": timestamp_ms,
    }
    results_dir.joinpath(f"{result_uuid}-result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


def _write_command_output(
    terminalreporter: pytest.TerminalReporter,
    result: subprocess.CompletedProcess[str],
) -> None:
    if result.stdout.strip():
        terminalreporter.write_line(result.stdout.rstrip())
    if result.stderr.strip():
        terminalreporter.write_line(result.stderr.rstrip(), red=True)


def _allure_command(root: Path) -> list[str] | None:
    project_allure = root / "node_modules" / ".bin" / "allure"
    if project_allure.is_file():
        bun = shutil.which("bun")
        if shutil.which("node") is None and bun is not None:
            return [bun, str(project_allure)]
        return [str(project_allure)]
    found = shutil.which("allure")
    return [found] if found else None


def pytest_sessionstart(session: pytest.Session) -> None:
    if hasattr(session.config, "workerinput"):
        return

    root = Path(session.config.rootpath)
    for path in [ALLURE_REPORT_DIR, COVERAGE_JSON, COVERAGE_MARKDOWN]:
        _remove_path(root / path)


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: pytest.ExitCode,
    config: pytest.Config,
) -> t.Generator[None]:
    yield

    if hasattr(config, "workerinput"):
        return

    root = Path(config.rootpath)
    allure_cmd = _allure_command(root)
    if allure_cmd is None:
        terminalreporter.write_sep("=", "ALLURE REPORT", red=True, bold=True)
        terminalreporter.write_line(
            "Allure CLI not found; source tools/dev to install node_modules/.bin/allure.",
            red=True,
        )
        return

    results_dir = root / ALLURE_RESULTS_DIR
    if not _has_allure_results(results_dir):
        terminalreporter.write_sep("=", "ALLURE REPORT", yellow=True, bold=True)
        terminalreporter.write_line("No Allure results were generated.", yellow=True)
        return

    _write_coverage_result(root)
    _remove_path(root / ALLURE_REPORT_DIR)
    result = subprocess.run(
        [
            *allure_cmd,
            "generate",
            "reports/allure-results",
            "--output",
            "reports/allure-report",
            "--name",
            "deap-er Pytest",
        ],
        cwd=config.rootpath,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        terminalreporter.write_sep("=", "ALLURE REPORT FAILED", red=True, bold=True)
        _write_command_output(terminalreporter, result)
        return

    terminalreporter.write_sep("=", "ALLURE REPORT", green=True, bold=True)
    terminalreporter.write_line("Allure report generated at reports/allure-report/index.html")
