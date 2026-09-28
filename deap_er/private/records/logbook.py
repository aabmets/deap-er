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
from collections import defaultdict
from typing import Any, override

from .logbook_format import format_txt
from .logbook_rows import LogbookRows
from .record_json import json_ready

__all__: list[str] = ["Logbook"]


class Logbook(LogbookRows):
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
        self.header_streamed: bool = False
        super().__init__()

    @property
    def stream(self) -> str:
        """Formatted text of entries recorded since the last stream read."""
        start_index, self.buff_index = self.buff_index, len(self)
        include_header = self.log_header and not self.header_streamed
        lines = format_txt(self, start_index, include_header=include_header)
        if include_header and (self.header or self):
            self.header_streamed = True
        return "\n".join(lines)

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

    def __txt__(self, start_index: int) -> list[str]:
        """Format rows from ``start_index`` as aligned column strings.

        Args:
            start_index: First entry to include.

        Returns:
            One formatted line per row, including a header when
            ``start_index`` is 0 and ``log_header`` is True.
        """
        return format_txt(self, start_index)

    @override
    def __str__(self) -> str:
        """Return the logbook as an aligned text table."""
        return "\n".join(self.__txt__(0))

    def to_json(self) -> str:
        """Serialize entries, chapters, and the header to JSON.

        NumPy scalars become Python numbers. Other non-JSON values
        become strings.

        Returns:
            A JSON document.
        """
        payload = {
            "header": self.header,
            "entries": [json_ready(entry) for entry in self],
            "chapters": {
                name: json.loads(chapter.to_json()) for name, chapter in self.chapters.items()
            },
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, text: str) -> "Logbook":
        """Rebuild a logbook from :meth:`to_json` output.

        Args:
            text: JSON document produced by :meth:`to_json`.

        Returns:
            A logbook with restored entries, chapters, and header.
        """
        data = json.loads(text)
        book = cls()
        book.header = list(data.get("header", []))
        book.extend(data.get("entries", []))
        for name, chapter in data.get("chapters", {}).items():
            book.chapters[name] = cls.from_json(json.dumps(chapter))
        return book
