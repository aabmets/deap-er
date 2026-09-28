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
from typing import Any, Self, SupportsIndex, override

__all__: list[str] = ["LogbookRows"]

_REJECT = "Logbook entries are chronological; {op} is not supported, use {alt}"


class LogbookRows(list[dict[str, Any]]):
    """List of logbook entries kept in sync with chapters and the stream.

    Removals drop the paired chapter rows and move the stream cursor.
    Operations that would reorder or rewrite ranges of the chronology
    raise ``TypeError``. Appending and extending add entries at the end.
    """

    chapters: dict[str, Any]
    buff_index: int

    @override
    def pop(self, index: SupportsIndex = 0) -> dict[str, Any]:
        """Remove and return the entry at ``index``.

        The stream cursor is moved back when the removed entry has
        already been streamed. The chapter row that shares ``gen``
        is removed from every chapter. A row without ``gen`` is
        paired by index when the chapter is the same length.

        Args:
            index: Position of the entry to remove.

        Returns:
            The removed entry.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        idx = int(index)
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError("pop index out of range")
        generation = self[idx].get("gen")
        for chapter in self.chapters.values():
            if not chapter:
                continue
            match = self.chapter_index_for_generation(chapter, generation, idx)
            if match is not None:
                chapter.pop(match)
        if idx < self.buff_index:
            self.buff_index -= 1
        return super().pop(idx)

    @override
    def remove(self, value: dict[str, Any], /) -> None:
        """Remove the first entry equal to ``value``.

        Uses the same chapter pairing and stream-cursor rules as ``pop``.

        Args:
            value: Entry to remove.

        Raises:
            ValueError: If no entry equals ``value``.
        """
        self.pop(self.index(value))

    def _delete_slice(self, key: slice) -> None:
        """Delete a slice of entries and matching chapter rows.

        Args:
            key: Slice of entries to remove.
        """
        for i in sorted(range(*key.indices(len(self))), reverse=True):
            self.pop(i)

    def chapter_index_for_generation(
        self, chapter: "LogbookRows", generation: Any, parent_index: int
    ) -> int | None:
        """Return the chapter row that shares ``generation``.

        When several rows share a generation, the match is the
        occurrence that lines up with ``parent_index``. When
        ``generation`` is missing and the chapter is the same
        length as this logbook, the match is positional.

        Args:
            chapter: Nested logbook to search.
            generation: Generation value from the parent entry.
            parent_index: Parent row being paired.

        Returns:
            Matching chapter index, or None.
        """
        if generation is None:
            if 0 <= parent_index < len(chapter) == len(self):
                return parent_index
            return None
        remaining = sum(1 for entry in self[parent_index:] if entry.get("gen") == generation)
        matches = [i for i, entry in enumerate(chapter) if entry.get("gen") == generation]
        if remaining == 0 or len(matches) < remaining:
            return None
        return matches[-remaining]

    @override
    def __delitem__(self, key: SupportsIndex | slice, /) -> None:
        """Delete an entry and the same index from every chapter."""
        if isinstance(key, slice):
            self._delete_slice(key)
        else:
            self.pop(key)

    @override
    def clear(self) -> None:
        """Remove every entry and the matching chapter rows.

        Uses the same chapter pairing and stream-cursor rules as
        ``del logbook[:]``.
        """
        del self[:]

    @override
    def insert(self, index: SupportsIndex, value: dict[str, Any], /) -> None:
        """Insert ``value`` before ``index``.

        An entry inserted before the stream cursor counts as already
        streamed, so earlier rows are not streamed again.

        Args:
            index: Position to insert at, as for ``list.insert``.
            value: Entry to insert.
        """
        idx = int(index)
        if idx < 0:
            idx = max(idx + len(self), 0)
        if idx < self.buff_index:
            self.buff_index += 1
        super().insert(idx, value)

    @override
    def __setitem__(self, key: Any, value: Any, /) -> None:
        """Replace the entry at an integer index.

        Raises:
            TypeError: If ``key`` is a slice.
        """
        if isinstance(key, slice):
            raise TypeError(_REJECT.format(op="slice assignment", alt="del, insert or record"))
        super().__setitem__(key, value)

    @override
    def __imul__(self, value: SupportsIndex, /) -> Self:
        """Reject in-place repetition.

        Raises:
            TypeError: Always; use ``clear`` to empty the logbook.
        """
        raise TypeError(_REJECT.format(op="in-place repetition", alt="clear"))

    @override
    def sort(self, *args: Any, **kwargs: Any) -> None:
        """Reject reordering of chronological entries.

        Raises:
            TypeError: Always; sort a copy such as ``sorted(logbook)``.
        """
        raise TypeError(_REJECT.format(op="sort", alt="sorted(logbook)"))

    @override
    def reverse(self) -> None:
        """Reject reordering of chronological entries.

        Raises:
            TypeError: Always; iterate ``reversed(logbook)`` instead.
        """
        raise TypeError(_REJECT.format(op="reverse", alt="reversed(logbook)"))
