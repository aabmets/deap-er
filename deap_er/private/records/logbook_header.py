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
from typing import Any

__all__: list[str] = ["build_header"]


def _chapter_header_cells(
    name: str,
    chapter_lines: list[str],
    offset: int,
    n_lines: int,
    header: list[list[str]],
) -> None:
    """Fill header rows for a chapter column.

    Args:
        name: Chapter name, centred on the banner row.
        chapter_lines: Rendered lines of the chapter.
        offset: Header offset of the chapter.
        n_lines: Total number of banner rows.
        header: Banner rows to append cells to.
    """
    length = max(len(line.expandtabs()) for line in chapter_lines)
    blanks = n_lines - 2 - offset
    for i in range(blanks):
        header[i].append(" " * length)
    header[blanks].append(name.center(length))
    header[blanks + 1].append("-" * length)
    for i in range(offset):
        header[blanks + 2 + i].append(chapter_lines[i])


def _plain_header_cells(
    name: str,
    column_index: int,
    logbook: Any,
    str_matrix: list[list[str]],
    header: list[list[str]],
) -> None:
    """Fill header rows for a plain (non-chapter) column.

    Args:
        name: Column name, placed on the last banner row.
        column_index: Index of the column in each data row.
        logbook: Logbook used to size empty columns.
        str_matrix: Rendered data rows, used to size the column.
        header: Banner rows to append cells to.
    """
    if str_matrix:
        length = max(len(line[column_index].expandtabs()) for line in str_matrix)
    else:
        length = max(len(name), logbook.columns_len[column_index] if logbook.columns_len else 0)
    for line in header[:-1]:
        line.append(" " * length)
    header[-1].append(name)


def build_header(
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
        n_lines += max(offsets.values()) + 1
    header: list[list[str]] = [[] for _ in range(n_lines)]
    for j, name in enumerate(columns):
        if name in chapters_txt:
            _chapter_header_cells(name, chapters_txt[name], offsets[name], n_lines, header)
        else:
            _plain_header_cells(name, j, logbook, str_matrix, header)
    return header
