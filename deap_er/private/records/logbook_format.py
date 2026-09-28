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
from typing import Any, cast

from .logbook_header import build_header

__all__: list[str] = [
    "chapter_blocks",
    "build_rows",
    "default_columns",
    "format_txt",
]


def _chapter_row_for_parent(logbook: Any, chapter: Any, parent_index: int) -> dict[str, Any]:
    """Return the chapter entry paired with a parent row, or ``{}``.

    Pair by ``gen`` when present; otherwise positional if lengths match.

    Args:
        logbook: Parent logbook that owns ``parent_index``.
        chapter: Nested logbook to read.
        parent_index: Parent row being rendered.
    """
    generation = logbook[parent_index].get("gen")
    match = logbook.chapter_index_for_generation(chapter, generation, parent_index)
    return {} if match is None else chapter[match]


class _AlignedChapter:
    """Chapter rows lined up with a parent logbook."""

    def __init__(self, logbook: Any, chapter: Any) -> None:
        self._chapter = chapter
        self.header = chapter.header or (default_columns(chapter) if len(chapter) else [])
        self.log_header = chapter.log_header
        self._rows = [_chapter_row_for_parent(logbook, chapter, i) for i in range(len(logbook))]
        self.chapters = {
            name: _AlignedChapter(logbook, nested) for name, nested in chapter.chapters.items()
        }

    @property
    def columns_len(self) -> list[int]:
        return cast(list[int], self._chapter.columns_len)

    @columns_len.setter
    def columns_len(self, value: list[int]) -> None:
        self._chapter.columns_len = value

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key: Any) -> Any:
        return self._rows[key]


def default_columns(logbook: Any) -> list[str]:
    """Return the columns shown when ``logbook.header`` is empty.

    Args:
        logbook: Non-empty logbook or aligned chapter.

    Returns:
        Sorted keys of the first entry, then sorted chapter names.
    """
    return cast(list[str], sorted(logbook[0].keys()) + sorted(logbook.chapters.keys()))


def chapter_blocks(
    logbook: Any, start_index: int, include_header: bool
) -> tuple[dict[str, list[str]], defaultdict[str, int]]:
    """Render every chapter and measure how far each one leads.

    A chapter carries its own header rows, so it produces more
    lines than this logbook renders entries. That surplus is the
    chapter's offset.

    Args:
        logbook: Logbook whose chapters to render.
        start_index: First entry to include.
        include_header: Whether the parent banner is rendered, so
            chapters render their own header rows too.

    Returns:
        The rendered lines of each chapter and their offsets.
    """
    chapters_txt: dict[str, list[str]] = {}
    offsets: defaultdict[str, int] = defaultdict(int)
    n_rows = len(logbook) - start_index
    for name, chapter in logbook.chapters.items():
        if isinstance(chapter, _AlignedChapter):
            view = chapter
        else:
            view = _AlignedChapter(logbook, chapter)
        chapter_header = include_header and view.log_header
        chapters_txt[name] = format_txt(view, start_index, include_header=chapter_header)
        offsets[name] = len(chapters_txt[name]) - n_rows
    return chapters_txt, offsets


def build_rows(
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


def format_txt(logbook: Any, start_index: int, include_header: bool | None = None) -> list[str]:
    """Format rows from ``start_index`` as aligned column strings.

    Args:
        logbook: Logbook to format.
        start_index: First entry to include.
        include_header: If True, prepend the banner. ``None`` means
            include it when ``start_index`` is 0 and ``log_header``
            is True.

    Returns:
        One formatted line per row, including a header when requested.
    """
    if include_header is None:
        include_header = bool(start_index == 0 and logbook.log_header)
    columns = logbook.header
    if not len(logbook):
        if columns and include_header:
            if not logbook.columns_len or len(logbook.columns_len) != len(columns):
                logbook.columns_len = list(map(len, columns))
            template = "\t".join(
                f"{{{i}:<{length}}}" for i, length in enumerate(logbook.columns_len)
            )
            return [template.format(*columns)]
        return ["The Logbook is empty."]
    if not columns:
        columns = default_columns(logbook)
    if not logbook.columns_len or len(logbook.columns_len) != len(columns):
        logbook.columns_len = list(map(len, columns))

    chapters_txt, offsets = chapter_blocks(logbook, start_index, include_header)
    str_matrix = build_rows(logbook, columns, start_index, chapters_txt, offsets)

    rows: Iterable[list[str]] = str_matrix
    if include_header:
        header = build_header(logbook, columns, chapters_txt, offsets, str_matrix)
        rows = chain(header, str_matrix)

    template = "\t".join(f"{{{i}:<{length}}}" for i, length in enumerate(logbook.columns_len))
    return [template.format(*line) for line in rows]
