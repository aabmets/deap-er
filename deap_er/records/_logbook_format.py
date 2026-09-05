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
from collections import defaultdict
from collections.abc import Iterable
from itertools import chain
from typing import Any

__all__: list[str] = []


def _chapter_blocks(
    logbook: Any, start_index: int
) -> tuple[dict[str, list[str]], defaultdict[str, int]]:
    """Render every chapter and measure how far each one leads.

    A chapter carries its own header rows, so it produces more
    lines than this logbook has entries. That surplus is the
    chapter's offset.

    Args:
        logbook: Logbook whose chapters to render.
        start_index: First entry to include.

    Returns:
        The rendered lines of each chapter and their offsets.
    """
    chapters_txt: dict[str, list[str]] = {}
    offsets: defaultdict[str, int] = defaultdict(int)
    for name, chapter in logbook.chapters.items():
        chapters_txt[name] = chapter.__txt__(start_index)
        if start_index == 0:
            offsets[name] = len(chapters_txt[name]) - len(logbook)
    return chapters_txt, offsets


def _build_rows(
    logbook: Any,
    columns: list[str],
    start_index: int,
    chapters_txt: dict[str, list[str]],
    offsets: defaultdict[str, int],
) -> list[list[str]]:
    """Render the data rows and widen ``columns_len`` to fit them.

    Args:
        logbook: Logbook whose entries to render.
        columns: Column names, in display order.
        start_index: First entry to include.
        chapters_txt: Rendered lines of each chapter.
        offsets: Header offset of each chapter.

    Returns:
        One list of cell strings per row.
    """
    str_matrix: list[list[str]] = []
    for i, line in enumerate(logbook[start_index:]):
        str_line: list[str] = []
        for j, name in enumerate(columns):
            if name in chapters_txt:
                column = chapters_txt[name][i + offsets[name]]
            else:
                value = line.get(name, "")
                string = "{0:n}" if isinstance(value, float) else "{0}"
                column = string.format(value)
            logbook.columns_len[j] = max(logbook.columns_len[j], len(column))
            str_line.append(column)
        str_matrix.append(str_line)
    return str_matrix


def _build_header(
    logbook: Any,
    columns: list[str],
    chapters_txt: dict[str, list[str]],
    offsets: defaultdict[str, int],
    str_matrix: list[list[str]],
) -> list[list[str]]:
    """Build the banner rows that sit above the data rows.

    Chapter names are centred over their columns above a dashed
    rule; plain columns are named on the last row only.

    Args:
        logbook: Logbook whose header to build.
        columns: Column names, in display order.
        chapters_txt: Rendered lines of each chapter.
        offsets: Header offset of each chapter.
        str_matrix: Rendered data rows, used to size plain columns.

    Returns:
        One list of cell strings per header row.
    """
    n_lines = 1
    if len(logbook.chapters) > 0:
        n_lines += max(map(len, chapters_txt.values())) - len(logbook) + 1
    header: list[list[str]] = [[] for _ in range(n_lines)]
    for j, name in enumerate(columns):
        if name in chapters_txt:
            length = max(len(line.expandtabs()) for line in chapters_txt[name])
            blanks = n_lines - 2 - offsets[name]
            for i in range(blanks):
                header[i].append(" " * length)
            header[blanks].append(name.center(length))
            header[blanks + 1].append("-" * length)
            for i in range(offsets[name]):
                header[blanks + 2 + i].append(chapters_txt[name][i])
        else:
            length = max(len(line[j].expandtabs()) for line in str_matrix)
            for line in header[:-1]:
                line.append(" " * length)
            header[-1].append(name)
    return header


def _format_txt(logbook: Any, start_index: int) -> list[str]:
    """Format rows from ``start_index`` as aligned column strings.

    Args:
        logbook: Logbook to format.
        start_index: First entry to include.

    Returns:
        One formatted line per row, including a header when
        ``start_index`` is 0 and ``log_header`` is True.
    """
    columns = logbook.header
    if not len(logbook):
        return ["The Logbook is empty."]
    if not columns:
        columns = sorted(logbook[0].keys()) + sorted(logbook.chapters.keys())
    if not logbook.columns_len or len(logbook.columns_len) != len(columns):
        logbook.columns_len = list(map(len, columns))

    chapters_txt, offsets = _chapter_blocks(logbook, start_index)
    str_matrix = _build_rows(logbook, columns, start_index, chapters_txt, offsets)

    rows: Iterable[list[str]] = str_matrix
    if start_index == 0 and logbook.log_header:
        header = _build_header(logbook, columns, chapters_txt, offsets, str_matrix)
        rows = chain(header, str_matrix)

    template = "\t".join(f"{{{i}:<{length}}}" for i, length in enumerate(logbook.columns_len))
    return [template.format(*line) for line in rows]
