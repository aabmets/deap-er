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
import itertools
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

from deap_er.rng import rng

from ._moving_peaks_catalog import MPConfigs, MPFuncs
from ._moving_peaks_shift import change_peaks as _change_peaks

__all__ = ["MovingPeaks", "MPConfigs", "MPFuncs"]

type PeakFunc = Callable[[Sequence[float], Iterable[float], float, float], float]


class MovingPeaks:
    """A fitness landscape whose peaks change over time.

    Peaks move in height, width, and location. If ``npeaks`` is a list
    of three integers, the peak count fluctuates between the first and
    third values, starting at the second. Fluctuating the count requires
    ``change_severity`` in kwargs. The default preset is
    ``MPConfigs.DEFAULT``.

    .. dropdown:: Table of Kwargs
       :margin: 0 5 0 0

       =================== ========== =================================================================================
       Parameter           Type       Details
       =================== ========== =================================================================================
       ``pfunc``           *Callable* The peak function or a list of peak functions.
       ``bfunc``           *Callable* Basis function for static landscape.
       ``npeaks``          *NumOrSeq* Number of peaks. An integer or a list of three integers [min, initial, max].
       ``change_severity`` *float*    The fraction of the number of peaks that is allowed to change.
       ``min_coord``       *float*    Minimum coordinate for the centre of the peaks.
       ``max_coord``       *float*    Maximum coordinate for the centre of the peaks.
       ``min_height``      *float*    Minimum height of the peaks.
       ``max_height``      *float*    Maximum height of the peaks.
       ``uniform_height``  *float*    Starting height of all peaks. Random, if ``uniform_height <= 0``.
       ``min_width``       *float*    Minimum width of the peaks.
       ``max_width``       *float*    Maximum width of the peaks
       ``uniform_width``   *float*    Starting width of all peaks. Random, if ``uniform_width <= 0``.
       ``lambda_``          *float*    Correlation between changes.
       ``move_severity``   *float*    The distance a single peak moves when peaks change.
       ``height_severity`` *float*    The standard deviation of the change to the height of a peak when peaks change.
       ``width_severity``  *float*    The standard deviation of the change to the width of a peak when peaks change.
       ``period``          *int*      Period between two changes.
       =================== ========== =================================================================================
    """

    def __init__(self, dimensions: int, **kwargs: Any) -> None:
        """Build a moving-peaks landscape.

        Args:
            dimensions: Dimensionality of the search domain.
            **kwargs: Optional landscape settings. See the class
                docstring table of kwargs.
        """
        self.dim = dimensions
        sc: dict[str, Any] = dict(MPConfigs.DEFAULT)
        sc.update(kwargs)

        n_peaks_val = cast(int | Sequence[int], sc["npeaks"])
        pfunc = cast(PeakFunc | Sequence[PeakFunc], sc["pfunc"])

        self.min_peaks: int | None = None
        self.max_peaks: int | None = None
        self.number_severity: float = 0.0
        if isinstance(n_peaks_val, Sequence) and not isinstance(n_peaks_val, (str, bytes)):
            self.min_peaks, n_peaks, self.max_peaks = n_peaks_val
            severity = sc["change_severity"]
            self.number_severity = 0.0 if severity is None else float(severity)
        else:
            n_peaks = int(n_peaks_val)

        if isinstance(pfunc, Sequence):
            funcs = list(pfunc)
            if len(funcs) == n_peaks:
                self.peaks_function = funcs
            else:
                self.peaks_function = rng.sample(funcs, n_peaks)
            self.pfunc_pool: tuple[PeakFunc, ...] = tuple(funcs)
        else:
            self.peaks_function = list(itertools.repeat(pfunc, n_peaks))
            self.pfunc_pool = (pfunc,)

        self.last_change_vector = [
            [rng.random() - 0.5 for _ in range(dimensions)] for _ in range(n_peaks)
        ]
        self.min_coord = float(sc["min_coord"])
        self.max_coord = float(sc["max_coord"])
        self.peaks_position = [
            [rng.uniform(self.min_coord, self.max_coord) for _ in range(dimensions)]
            for _ in range(n_peaks)
        ]
        uniform_height = float(sc["uniform_height"])
        self.min_height = float(sc["min_height"])
        self.max_height = float(sc["max_height"])
        if uniform_height != 0:
            self.peaks_height = [uniform_height for _ in range(n_peaks)]
        else:
            self.peaks_height = [
                rng.uniform(self.min_height, self.max_height) for _ in range(n_peaks)
            ]

        uniform_width = float(sc["uniform_width"])
        self.min_width = float(sc["min_width"])
        self.max_width = float(sc["max_width"])
        if uniform_width != 0:
            self.peaks_width = [uniform_width for _ in range(n_peaks)]
        else:
            self.peaks_width = [rng.uniform(self.min_width, self.max_width) for _ in range(n_peaks)]

        self.basis_function: Callable[[Sequence[float]], float] | None = sc.get("bfunc")
        self.move_severity = float(sc["move_severity"])
        self.height_severity = float(sc["height_severity"])
        self.width_severity = float(sc["width_severity"])
        self.period = int(sc["period"])
        self.lamb = float(sc["lambda_"])
        self._optimum: float | None = None
        self._error: float | None = None
        self._offline_error = 0.0
        self.nevals = 0

    def __call__(self, individual: Sequence[float], count: bool = True) -> tuple[float]:
        """Evaluate the given **individual** in the context of the current configuration.

        Args:
            individual: Individual to evaluate.
            count: Whether to include this evaluation in the
                evaluation count and error statistics.

        Returns:
            The fitness of the individual.
        """
        possible_values = []
        zipper = zip(
            self.peaks_function,
            self.peaks_position,
            self.peaks_height,
            self.peaks_width,
            strict=False,
        )
        for func, pos, height, width in zipper:
            result = func(individual, pos, height, width)
            possible_values.append(result)

        if self.basis_function:
            result = self.basis_function(individual)
            possible_values.append(result)

        fitness = max(possible_values)

        if count:
            self.nevals += 1
            if self._optimum is None or self._error is None:
                self._optimum = self.global_maximum[0]
                self._error = abs(fitness - self._optimum)
            else:
                self._error = min(self._error, abs(fitness - self._optimum))
            self._offline_error += self._error

            if self.period > 0 and self.nevals % self.period == 0:
                self.change_peaks()

        return (float(fitness),)

    @property
    def global_maximum(self) -> tuple[float, list[float]]:
        """Returns the value and position of the largest peak."""
        potential_max = []
        zipper = zip(
            self.peaks_function,
            self.peaks_position,
            self.peaks_height,
            self.peaks_width,
            strict=False,
        )
        for func, pos, height, width in zipper:
            result = func(pos, pos, height, width)
            value: tuple[float, list[float]] = (float(result), pos)
            potential_max.append(value)
        return max(potential_max)

    @property
    def sorted_maxima(self) -> list[tuple[float, list[float]]]:
        """Return visible peak values and positions, largest first."""
        maximums = []
        zipper = zip(
            self.peaks_function,
            self.peaks_position,
            self.peaks_height,
            self.peaks_width,
            strict=False,
        )
        for func, pos, height, width in zipper:
            result = func(pos, pos, height, width)
            if result >= self.__call__(pos, count=False)[0]:
                value: tuple[float, list[float]] = (float(result), pos)
                maximums.append(value)
        return sorted(maximums, reverse=True)

    @property
    def offline_error(self) -> float:
        """Returns the offline error of the landscape, or 0.0 before the first evaluation."""
        if not self.nevals:
            return 0.0
        return float(self._offline_error / self.nevals)

    @property
    def current_error(self) -> float | None:
        """Returns the current error of the landscape."""
        return self._error

    def change_peaks(self) -> None:
        """Changes the position, the height, the width and the number of peaks."""
        _change_peaks(self)
