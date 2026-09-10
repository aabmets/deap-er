#!/usr/bin/env python3
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
"""Validate and promote Keep a Changelog Unreleased notes for a release."""

import argparse
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHANGELOG = _REPO_ROOT / "CHANGELOG.md"

_UNRELEASED_H2 = re.compile(r"^## \[Unreleased\][ \t]*$", re.MULTILINE)
_VERSION_H2 = re.compile(
    r"^## \[v?(\d+\.\d+\.\d+(?:[.-][0-9A-Za-z.]+)?)\](?:[ \t]+-[ \t]+\S+)?[ \t]*$",
    re.MULTILINE,
)
_FOOTER_UNRELEASED = re.compile(
    r"^\[Unreleased\]:\s*(https://github\.com/[^/\s]+/[^/\s]+)/compare/"
    r"(\S+)\.\.\.HEAD[ \t]*$",
    re.MULTILINE,
)
_FOOTER_VERSION = re.compile(
    r"^\[v?(\d+\.\d+\.\d+(?:[.-][0-9A-Za-z.]+)?)\]:\s*\S+[ \t]*$",
    re.MULTILINE,
)
_ANY_GITHUB_LINK = re.compile(
    r"^\[.+?\]:\s*(https://github\.com/[^/\s]+/[^/\s]+)/",
    re.MULTILINE,
)


def _fail(message: str) -> NoReturn:
    """Exit with ``message`` on stderr."""
    sys.exit(f"Error: {message}")


def _new_version() -> str:
    """Return ``NEW_VERSION`` without a leading ``v``."""
    raw = os.environ.get("NEW_VERSION", "").strip()
    if not raw:
        _fail("NEW_VERSION is not set")
    return raw.removeprefix("v")


def _read_changelog() -> str:
    """Return the changelog text."""
    if not _CHANGELOG.is_file():
        _fail(f"changelog not found: {_CHANGELOG}")
    return _CHANGELOG.read_text(encoding="utf-8")


def _unreleased_heading(text: str) -> re.Match[str]:
    """Return the ``## [Unreleased]`` match."""
    match = _UNRELEASED_H2.search(text)
    if match is None:
        _fail("CHANGELOG.md has no '## [Unreleased]' heading")
    return match


def _unreleased_body(text: str) -> str:
    """Return text between Unreleased and the next H2 heading."""
    match = _unreleased_heading(text)
    body_start = match.end()
    next_h2 = re.search(r"^## \[", text[body_start:], re.MULTILINE)
    body_end = body_start + next_h2.start() if next_h2 else len(text)
    return text[body_start:body_end]


def _version_headings(text: str) -> list[str]:
    """Return version strings from ``## [x.y.z]`` headings, newest first."""
    return [match.group(1) for match in _VERSION_H2.finditer(text)]


def _footer_versions(text: str) -> list[str]:
    """Return version strings already linked in the changelog footer."""
    return [match.group(1) for match in _FOOTER_VERSION.finditer(text)]


def _require_unreleased_content(text: str) -> None:
    """Exit if the Unreleased section has no notes."""
    if not _unreleased_body(text).strip():
        _fail("CHANGELOG.md [Unreleased] has no notes to release")


def _require_version_absent(text: str, version: str) -> None:
    """Exit if ``version`` is already a heading or footer link."""
    if version in _version_headings(text):
        _fail(f"CHANGELOG.md already has a heading for {version}")
    if version in _footer_versions(text):
        _fail(f"CHANGELOG.md footer already has a [{version}] link")


def _previous_version(text: str) -> str:
    """Return the newest versioned heading (the current latest release)."""
    versions = _version_headings(text)
    if not versions:
        _fail("CHANGELOG.md has no previous version heading for compare links")
    return versions[0]


def _repo_base(text: str) -> str:
    """Return ``https://github.com/owner/repo`` from footer links."""
    match = _FOOTER_UNRELEASED.search(text)
    if match is not None:
        return match.group(1)
    any_link = _ANY_GITHUB_LINK.search(text)
    if any_link is not None:
        return any_link.group(1)
    _fail("CHANGELOG.md footer has no GitHub compare links")


def _line_end(text: str, match: re.Match[str]) -> int:
    """Return the index after ``match``, including a trailing newline."""
    end = match.end()
    if end < len(text) and text[end] == "\n":
        return end + 1
    return end


def _insert_release_heading(text: str, version: str, today: str) -> str:
    """Insert ``## [version] - date`` immediately under Unreleased."""
    line_end = _line_end(text, _unreleased_heading(text))
    heading = f"\n## [{version}] - {today}\n"
    return f"{text[:line_end]}{heading}{text[line_end:]}"


def _rewrite_footer(text: str, new_version: str, previous: str) -> str:
    """Point Unreleased at ``new_version`` and add its compare link."""
    base = _repo_base(text)
    unreleased_line = f"[Unreleased]: {base}/compare/{new_version}...HEAD"
    version_line = f"[{new_version}]: {base}/compare/{previous}...{new_version}"
    if _FOOTER_UNRELEASED.search(text):
        text = _FOOTER_UNRELEASED.sub(unreleased_line, text, count=1)
    else:
        text = text.rstrip() + f"\n\n{unreleased_line}\n"
    unreleased_match = _FOOTER_UNRELEASED.search(text)
    if unreleased_match is None:
        return text.rstrip() + f"\n{version_line}\n"
    insert_at = _line_end(text, unreleased_match)
    return f"{text[:insert_at]}{version_line}\n{text[insert_at:]}"


def cmd_check() -> None:
    """Exit non-zero unless Unreleased notes can be released as ``NEW_VERSION``."""
    version = _new_version()
    text = _read_changelog()
    _require_version_absent(text, version)
    _require_unreleased_content(text)
    _previous_version(text)
    _repo_base(text)
    print(f"CHANGELOG.md is ready to release {version}")


def cmd_promote() -> None:
    """Move Unreleased notes under ``NEW_VERSION`` and fix footer links."""
    version = _new_version()
    text = _read_changelog()
    _require_version_absent(text, version)
    _require_unreleased_content(text)
    previous = _previous_version(text)
    today = datetime.now(UTC).date().isoformat()
    text = _insert_release_heading(text, version, today)
    text = _rewrite_footer(text, version, previous)
    _CHANGELOG.write_text(text, encoding="utf-8")
    print(f"Promoted [Unreleased] notes to {version} (previous {previous})")


def cmd_verify() -> None:
    """Exit non-zero unless the first versioned H2 is ``NEW_VERSION``."""
    version = _new_version()
    headings = _version_headings(_read_changelog())
    if not headings:
        _fail("CHANGELOG.md has no version heading")
    if headings[0] != version:
        _fail(f"CHANGELOG.md first version heading is {headings[0]}, expected {version}")
    print(f"CHANGELOG.md first version heading is {version}")


def main() -> None:
    """Dispatch ``check``, ``promote``, or ``verify``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "promote", "verify"))
    args = parser.parse_args()
    {"check": cmd_check, "promote": cmd_promote, "verify": cmd_verify}[args.command]()


if __name__ == "__main__":
    main()
