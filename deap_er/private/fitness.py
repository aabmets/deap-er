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
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, SupportsFloat, cast, override

__all__: list[str] = ["FitnessValues", "Fitness"]

type FitnessValues = SupportsFloat | Iterable[SupportsFloat]
"""A single objective value or an iterable of them, including NumPy scalars and arrays.

:meta private:
"""


class Fitness:
    """Quality of a solution, compared through weighted objectives.

    The class attribute ``weights`` must be set before a Fitness object
    can be instantiated. A fitness may be created without values, but
    it stays invalid until ``values`` is assigned a sequence of the
    same length as ``weights``.

    Args:
        values: Initial objective values. Optional.

    Attributes:
        weights: Shared per fitness type. Each element is a real number
            for one objective: negative means minimize, positive means
            maximize.
    """

    weights: Sequence[int | float] = ()
    wvalues: tuple[float, ...] = ()
    _values: tuple[float, ...] = ()
    crowding_dist: float = 0.0

    def __init__(self, values: FitnessValues | None = None) -> None:
        """See the class docstring."""
        if not self.weights:
            raise TypeError(
                "Can't instantiate 'Fitness', when class attribute 'weights' tuple is not set."
            )
        if values is not None:
            self.values = values

    @property
    def values(self) -> tuple[float, ...]:
        """Objective values of the individual.

        The setter accepts a number or a sequence of numbers. A single
        number is stored as a one-element sequence. The getter returns
        a tuple of floats, or an empty tuple when the fitness is
        invalid. Deleting the property clears the stored values.

        Raises:
            TypeError: If the assigned sequence length does not match
                ``weights``.
        """
        if self.is_valid():
            return self._values
        return ()

    @values.setter
    def values(self, values: FitnessValues) -> None:
        if isinstance(values, tuple):
            raw: tuple[Any, ...] = values
        elif isinstance(values, Iterable):
            raw = tuple(values)
        else:
            raw = (values,)
        if len(raw) != len(self.weights):
            raise TypeError(
                "The assigned values must have the same length as "
                "the 'weights' attribute of the 'Fitness' class."
            )
        if raw and type(raw[0]) is float:
            seq = cast(tuple[float, ...], raw)
        else:
            seq = tuple(float(value) for value in raw)
        self._values = seq
        weights = self.weights
        self.wvalues = tuple(value * weight for value, weight in zip(seq, weights, strict=True))

    @values.deleter
    def values(self) -> None:
        self._values = ()
        self.wvalues = ()

    def dominates(self, other: Fitness, slc: slice | None = None) -> bool:
        """Return whether this fitness Pareto-dominates ``other``.

        Each compared objective of ``self`` must be at least as good as
        the corresponding objective of ``other``, and at least one must
        be strictly better.

        Args:
            other: Fitness to test against.
            slc: Slice of objectives to compare. Optional; all
                objectives are used when omitted.

        Returns:
            True if ``self`` dominates ``other``.
        """
        own = self.wvalues if slc is None else self.wvalues[slc]
        theirs = other.wvalues if slc is None else other.wvalues[slc]
        better = False
        for a, b in zip(own, theirs, strict=True):
            if a < b:
                return False
            if a > b:
                better = True
        return better

    def is_valid(self) -> bool:
        """Return whether this fitness has a complete set of values.

        Returns:
            True if ``weights`` is non-empty and ``values`` has the
            same length.
        """
        return len(self.wvalues) == len(self.weights) > 0

    def __gt__(self, other: Fitness) -> bool:
        """Return whether this fitness is strictly better than ``other``."""
        return self.wvalues > other.wvalues

    def __ge__(self, other: Fitness) -> bool:
        """Return whether this fitness is at least as good as ``other``."""
        return self.wvalues >= other.wvalues

    def __le__(self, other: Fitness) -> bool:
        """Return whether this fitness is at most as good as ``other``."""
        return self.wvalues <= other.wvalues

    def __lt__(self, other: Fitness) -> bool:
        """Return whether this fitness is strictly worse than ``other``."""
        return self.wvalues < other.wvalues

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether the two fitnesses compare equal."""
        if not isinstance(other, Fitness):
            return NotImplemented
        return self.wvalues == other.wvalues

    @override
    def __ne__(self, other: object) -> bool:
        """Return whether the two fitnesses compare unequal."""
        if not isinstance(other, Fitness):
            return NotImplemented
        return self.wvalues != other.wvalues

    def __len__(self) -> int:
        """Return the number of stored weighted values."""
        return len(self.wvalues)

    @override
    def __hash__(self) -> int:
        """Hash the weighted values."""
        return hash(self.wvalues)

    @override
    def __str__(self) -> str:
        """Return the unweighted values as a string."""
        return str(self.values)

    @override
    def __repr__(self) -> str:
        """Return a reconstructable representation."""
        return f"{self.__module__}.{self.__class__.__name__}({str(self.values)})"

    def __deepcopy__(self, memo: dict[int, Any]) -> Fitness:
        """Return a new Fitness with the same weighted values."""
        copy = self.__class__()
        copy.wvalues = self.wvalues
        copy._values = self._values
        return copy
