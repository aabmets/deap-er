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

__all__ = [
    "EXAMPLES_MARK",
    "exclusive_examples",
    "pytest_collection_modifyitems",
    "pytest_configure",
]

EXAMPLES_MARK = "examples"


def exclusive_examples(config: pytest.Config) -> bool:
    """Return True when ``-m examples`` is the only mark filter.

    Args:
        config: Pytest config whose ``markexpr`` is inspected.

    Returns:
        True when the session should run example scripts and nothing else.
    """
    raw = getattr(config.option, "markexpr", "")
    if not isinstance(raw, str):
        return False
    return bool(raw.strip() == EXAMPLES_MARK)


def _disable_coverage(config: pytest.Config) -> None:
    if hasattr(config.option, "no_cov"):
        config.option.no_cov = True
    if hasattr(config.option, "cov_fail_under"):
        config.option.cov_fail_under = None
    cov = config.pluginmanager.get_plugin("_cov")
    if cov is None:
        return
    cov._disabled = True
    options = getattr(cov, "options", None)
    if options is None:
        return
    options.no_cov = True
    options.cov_fail_under = None


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "examples: docs/examples scripts; collected only with -m examples",
    )
    if exclusive_examples(config):
        _disable_coverage(config)


def _is_example(item: pytest.Item) -> bool:
    return item.get_closest_marker(EXAMPLES_MARK) is not None


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    keep_examples = exclusive_examples(config)
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        if _is_example(item) == keep_examples:
            selected.append(item)
        else:
            deselected.append(item)
    if not deselected:
        return
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
