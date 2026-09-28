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
import tomllib
from pathlib import Path

from tests.harness.allure_report_plugin import HTML_COVERAGE_INDEX

REPO = Path(__file__).resolve().parents[2]


def test_html_coverage_index_matches_the_configured_directory():
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    directory = Path(config["tool"]["coverage"]["html"]["directory"])

    assert directory / "index.html" == HTML_COVERAGE_INDEX
