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
from typing import Any, SupportsIndex, override

__all__ = ["Logbook"]


class Logbook(list[dict[str, Any]]):
    """Chronological evolution records as a list of dictionaries.

    Retrieve columns with ``select``. Nested dictionaries passed to
    ``record`` become named chapters. Set ``header`` to control column
    order when printing.
    """

    def __init__(self) -> None:
        """Create an empty logbook."""
        self.chapters = defaultdict(Logbook)
        self.buff_index: int = 0
        self.log_header: bool = True
        self.columns_len: list[int] = []
        self.header: list[str] = []
        super().__init__()

    @property
    def stream(self) -> str:
        """Formatted text of entries recorded since the last stream read."""
        start_index, self.buff_index = self.buff_index, len(self)
        return "\n".join(self.__txt__(start_index))

    def record(self, **data: Any) -> None:
        """Append one chronological entry.

        Nested dict values are recorded into named chapters. Remaining
        keys form the entry on this logbook. Non-dict keys are also
        copied into each chapter.

        Args:
            **data: Fields for the new entry. Dict values become chapters.
        """
        apply_to_all = {k: v for k, v in data.items() if not isinstance(v, dict)}
        for key, value in list(data.items()):
            if isinstance(value, dict):
                chapter_infos = value.copy()
                chapter_infos.update(apply_to_all)
                self.chapters[key].record(**chapter_infos)
                del data[key]
        self.append(data)

    def select(self, *names: str) -> list[Any]:
        """Return recorded values for one or more field names.

        A missing name yields ``None`` in that column. One name
        returns a flat list; several names return a list of lists.

        Args:
            *names: Field names to retrieve.

        Returns:
            Values for the requested names, in chronological order.
        """
        if len(names) == 1:
            return [entry.get(names[0], None) for entry in self]
        return [[entry.get(name, None) for entry in self] for name in names]

    @override
    def pop(self, index: SupportsIndex = 0) -> dict[str, Any]:
        """Remove and return the entry at ``index``.

        The stream cursor is moved back when the removed entry has
        already been streamed.

        Args:
            index: Position of the entry to remove.

        Returns:
            The removed entry.
        """
        idx = int(index)
        if idx < 0:
            idx += len(self)
        if idx < self.buff_index:
            self.buff_index -= 1
        return super().pop(idx)

    @override
    def __delitem__(self, key: SupportsIndex | slice, /) -> None:
        """Delete an entry and the same index from every chapter."""
        if isinstance(key, slice):
            for i in sorted(range(*key.indices(len(self))), reverse=True):
                self.pop(i)
                for chapter in self.chapters.values():
                    chapter.pop(i)
        else:
            idx = int(key)
            if idx < 0:
                idx += len(self)
            record = self[idx] if 0 <= idx < len(self) else {}
            generation = record.get("gen")
            self.pop(key)
            for chapter in self.chapters.values():
                if not chapter:
                    continue
                match = next(
                    (
                        i
                        for i, entry in enumerate(chapter)
                        if generation is not None and entry.get("gen") == generation
                    ),
                    None,
                )
                if match is not None:
                    chapter.pop(match)

    def _chapter_blocks(
        self, start_index: int
    ) -> tuple[dict[str, list[str]], defaultdict[str, int]]:
        """Render every chapter and measure how far each one leads.

        A chapter carries its own header rows, so it produces more
        lines than this logbook has entries. That surplus is the
        chapter's offset.

        Args:
            start_index: First entry to include.

        Returns:
            The rendered lines of each chapter and their offsets.
        """
        chapters_txt: dict[str, list[str]] = {}
        offsets: defaultdict[str, int] = defaultdict(int)
        for name, chapter in self.chapters.items():
            chapters_txt[name] = chapter.__txt__(start_index)
            if start_index == 0:
                offsets[name] = len(chapters_txt[name]) - len(self)
        return chapters_txt, offsets

    def _build_rows(
        self,
        columns: list[str],
        start_index: int,
        chapters_txt: dict[str, list[str]],
        offsets: defaultdict[str, int],
    ) -> list[list[str]]:
        """Render the data rows and widen ``columns_len`` to fit them.

        Args:
            columns: Column names, in display order.
            start_index: First entry to include.
            chapters_txt: Rendered lines of each chapter.
            offsets: Header offset of each chapter.

        Returns:
            One list of cell strings per row.
        """
        str_matrix: list[list[str]] = []
        for i, line in enumerate(self[start_index:]):
            str_line: list[str] = []
            for j, name in enumerate(columns):
                if name in chapters_txt:
                    column = chapters_txt[name][i + offsets[name]]
                else:
                    value = line.get(name, "")
                    string = "{0:n}" if isinstance(value, float) else "{0}"
                    column = string.format(value)
                self.columns_len[j] = max(self.columns_len[j], len(column))
                str_line.append(column)
            str_matrix.append(str_line)
        return str_matrix

    def _build_header(
        self,
        columns: list[str],
        chapters_txt: dict[str, list[str]],
        offsets: defaultdict[str, int],
        str_matrix: list[list[str]],
    ) -> list[list[str]]:
        """Build the banner rows that sit above the data rows.

        Chapter names are centred over their columns above a dashed
        rule; plain columns are named on the last row only.

        Args:
            columns: Column names, in display order.
            chapters_txt: Rendered lines of each chapter.
            offsets: Header offset of each chapter.
            str_matrix: Rendered data rows, used to size plain columns.

        Returns:
            One list of cell strings per header row.
        """
        n_lines = 1
        if len(self.chapters) > 0:
            n_lines += max(map(len, chapters_txt.values())) - len(self) + 1
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

    def __txt__(self, start_index: int) -> list[str]:
        """Format rows from ``start_index`` as aligned column strings.

        Args:
            start_index: First entry to include.

        Returns:
            One formatted line per row, including a header when
            ``start_index`` is 0 and ``log_header`` is True.
        """
        columns = self.header
        if not len(self):
            return ["The Logbook is empty."]
        if not columns:
            columns = sorted(self[0].keys()) + sorted(self.chapters.keys())
        if not self.columns_len or len(self.columns_len) != len(columns):
            self.columns_len = list(map(len, columns))

        chapters_txt, offsets = self._chapter_blocks(start_index)
        str_matrix = self._build_rows(columns, start_index, chapters_txt, offsets)

        rows: Iterable[list[str]] = str_matrix
        if start_index == 0 and self.log_header:
            header = self._build_header(columns, chapters_txt, offsets, str_matrix)
            rows = chain(header, str_matrix)

        template = "\t".join(f"{{{i}:<{length}}}" for i, length in enumerate(self.columns_len))
        return [template.format(*line) for line in rows]

    @override
    def __str__(self) -> str:
        """Return the logbook as an aligned text table."""
        return "\n".join(self.__txt__(0))
