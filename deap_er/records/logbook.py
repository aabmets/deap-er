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
from typing import Any, SupportsIndex, override

from ._logbook_format import _format_txt

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

    def _delete_slice(self, key: slice) -> None:
        """Delete a slice of entries and the same indexes from every chapter.

        Args:
            key: Slice of entries to remove.
        """
        for i in sorted(range(*key.indices(len(self))), reverse=True):
            self.pop(i)
            for chapter in self.chapters.values():
                chapter.pop(i)

    def _chapter_index_for_generation(self, chapter: "Logbook", generation: Any) -> int | None:
        """Return the chapter row that shares ``generation``.

        Args:
            chapter: Nested logbook to search.
            generation: Generation value from the parent entry.

        Returns:
            Matching chapter index, or None.
        """
        if generation is None:
            return None
        return next(
            (i for i, entry in enumerate(chapter) if entry.get("gen") == generation),
            None,
        )

    def _delete_index(self, key: SupportsIndex) -> None:
        """Delete one entry and the matching generation from every chapter.

        Args:
            key: Position of the entry to remove.
        """
        idx = int(key)
        if idx < 0:
            idx += len(self)
        record = self[idx] if 0 <= idx < len(self) else {}
        generation = record.get("gen")
        self.pop(key)
        for chapter in self.chapters.values():
            if not chapter:
                continue
            match = self._chapter_index_for_generation(chapter, generation)
            if match is not None:
                chapter.pop(match)

    @override
    def __delitem__(self, key: SupportsIndex | slice, /) -> None:
        """Delete an entry and the same index from every chapter."""
        if isinstance(key, slice):
            self._delete_slice(key)
        else:
            self._delete_index(key)

    def __txt__(self, start_index: int) -> list[str]:
        """Format rows from ``start_index`` as aligned column strings.

        Args:
            start_index: First entry to include.

        Returns:
            One formatted line per row, including a header when
            ``start_index`` is 0 and ``log_header`` is True.
        """
        return _format_txt(self, start_index)

    @override
    def __str__(self) -> str:
        """Return the logbook as an aligned text table."""
        return "\n".join(self.__txt__(0))
