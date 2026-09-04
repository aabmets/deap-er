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
import math
from collections.abc import Callable, Iterable, Sequence
from types import MappingProxyType
from typing import Any, cast

from deap_er.rng import rng

__all__ = ["MovingPeaks", "MPConfigs", "MPFuncs"]

type PeakFunc = Callable[[Sequence[float], Iterable[float], float, float], float]


def _change_shape(
    axis: list[float], axis_min: float, axis_max: float, sev: float, idx: int
) -> None:
    """Nudge one peak attribute, reflecting it off its bounds.

    Args:
        axis: Per-peak values, modified in place.
        axis_min: Lower bound of the attribute.
        axis_max: Upper bound of the attribute.
        sev: Standard deviation of the change.
        idx: Position of the peak to change.
    """
    change = rng.gauss(0, 1) * sev
    new_value = change + axis[idx]
    if new_value < axis_min:
        axis[idx] = 2.0 * axis_min - axis[idx] - change
    elif new_value > axis_max:
        axis[idx] = 2.0 * axis_max - axis[idx] - change
    else:
        axis[idx] = new_value


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

    def _remove_peaks(self, count: int) -> None:
        """Drop ``count`` randomly chosen peaks.

        Args:
            count: Number of peaks to remove.
        """
        for _ in range(count):
            idx = rng.randrange(len(self.peaks_function))
            self.peaks_function.pop(idx)
            self.peaks_position.pop(idx)
            self.peaks_height.pop(idx)
            self.peaks_width.pop(idx)
            self.last_change_vector.pop(idx)

    def _add_peaks(self, count: int) -> None:
        """Append ``count`` randomly placed peaks.

        Args:
            count: Number of peaks to add.
        """
        for _ in range(count):
            rand = rng.choice(self.pfunc_pool)
            self.peaks_function.append(rand)
            rand = [rng.uniform(self.min_coord, self.max_coord) for _ in range(self.dim)]
            self.peaks_position.append(rand)
            rand = rng.uniform(self.min_height, self.max_height)
            self.peaks_height.append(rand)
            rand = rng.uniform(self.min_width, self.max_width)
            self.peaks_width.append(rand)
            rand = [rng.random() - 0.5 for _ in range(self.dim)]
            self.last_change_vector.append(rand)

    def _change_peak_count(self) -> None:
        """Add or remove peaks, staying within the configured bounds."""
        if self.min_peaks is None or self.max_peaks is None:
            return

        n_peaks = len(self.peaks_function)
        u = rng.random()
        r = self.max_peaks - self.min_peaks
        if u < 0.5:
            u = rng.random()
            runs = int(round(r * u * self.number_severity))
            self._remove_peaks(min(n_peaks - self.min_peaks, runs))
        else:
            u = rng.random()
            runs = int(round(r * u * self.number_severity))
            self._add_peaks(min(self.max_peaks - n_peaks, runs))

    def _move_peak(self, index: int) -> None:
        """Shift one peak, reflecting it off the coordinate bounds.

        The step is a blend of a fresh random direction and the peak's
        previous direction, controlled by ``lambda_``.

        Args:
            index: Position of the peak to move.
        """
        len_ = len(self.peaks_position[index])
        shift = [rng.random() - 0.5 for _ in range(len_)]
        shift_length = sum(s**2 for s in shift)
        shift_length = self.move_severity / math.sqrt(shift_length) if shift_length > 0 else 0

        zipper = zip(shift, self.last_change_vector[index], strict=False)
        shift = [shift_length * (1.0 - self.lamb) * s + self.lamb * c for s, c in zipper]
        shift_length = sum(s**2 for s in shift)
        shift_length = self.move_severity / math.sqrt(shift_length) if shift_length > 0 else 0

        shift = [s * shift_length for s in shift]

        new_position = []
        final_shift = []
        for pp, s in zip(self.peaks_position[index], shift, strict=False):
            new_coord = pp + s
            if new_coord < self.min_coord:
                new_position.append(2.0 * self.min_coord - pp - s)
                final_shift.append(-1.0 * s)
            elif new_coord > self.max_coord:
                new_position.append(2.0 * self.max_coord - pp - s)
                final_shift.append(-1.0 * s)
            else:
                new_position.append(new_coord)
                final_shift.append(s)

        self.peaks_position[index] = new_position
        self.last_change_vector[index] = final_shift

    def change_peaks(self) -> None:
        """Changes the position, the height, the width and the number of peaks."""
        self._optimum = None

        self._change_peak_count()

        for i in range(len(self.peaks_function)):
            self._move_peak(i)
            _change_shape(
                self.peaks_height, self.min_height, self.max_height, self.height_severity, i
            )
            _change_shape(self.peaks_width, self.min_width, self.max_width, self.width_severity, i)


class MPFuncs:
    """Peak functions for Moving Peaks custom presets."""

    @staticmethod
    def pf1(
        individual: Sequence[float], positions: Iterable[float], height: float, width: float
    ) -> float:
        """The peak function of the :data:`DEFAULT` preset.

        Args:
            individual: Individual to evaluate.
            positions: Peak centre coordinates.
            height: Peak height.
            width: Peak width.

        Returns:
            The fitness of the individual.
        """
        value = 0.0
        for x, p in zip(individual, positions, strict=False):
            value += (x - p) ** 2
        return float(height / (1 + width * value))

    @staticmethod
    def pf2(
        individual: Sequence[float], positions: Iterable[float], height: float, width: float
    ) -> float:
        """The peak function of the :data:`ALT1` and :data:`ALT2` presets.

        Args:
            individual: Individual to evaluate.
            positions: Peak centre coordinates.
            height: Peak height.
            width: Peak width.

        Returns:
            The fitness of the individual.
        """
        value = 0.0
        for x, p in zip(individual, positions, strict=False):
            value += (x - p) ** 2
        return float(height - width * math.sqrt(value))

    @staticmethod
    def pf3(
        individual: Sequence[float], positions: Iterable[float], height: float, *_: Any
    ) -> float:
        """An optional peak function.

        Args:
            individual: Individual to evaluate.
            positions: Peak centre coordinates.
            height: Peak height.

        Returns:
            The fitness of the individual.
        """
        value = 0.0
        for x, p in zip(individual, positions, strict=False):
            value += (x - p) ** 2
        return float(height * value)


class MPConfigs:
    """Configuration presets for the Moving Peaks problem.

    Each preset is a ``dict`` class attribute.

    .. dropdown:: Table of Presets
       :margin: 0 5 0 0

        =================== ===================== ===================== =====================
        Keys / Presets      **DEFAULT**           **ALT1**              **ALT2**
        =================== ===================== ===================== =====================
        ``pfunc``           ``MPFuncs.pf1``       ``MPFuncs.pf2``       ``MPFuncs.pf2``
        ``bfunc``           :obj:`None`           :obj:`None`           :obj:`lambda x: 10`
        ``npeaks``          5                     10                    50
        ``change_severity`` :obj:`None`           :obj:`None`           :obj:`None`
        ``min_coord``       0.0                   0.0                   0.0
        ``max_coord``       100.0                 100.0                 100.0
        ``min_height``      30.0                  30.0                  30.0
        ``max_height``      70.0                  70.0                  70.0
        ``uniform_height``  50.0                  50.0                  0.0
        ``min_width``       0.0001                1.0                   1.0
        ``max_width``       0.2                   12.0                  12.0
        ``uniform_width``   0.1                   0.0                   0.0
        ``lambda_``         0.0                   0.5                   0.5
        ``move_severity``   1.0                   1.5                   1.0
        ``height_severity`` 7.0                   7.0                   1.0
        ``width_severity``  0.01                  1.0                   0.5
        ``period``          5000                  5000                  1000
        =================== ===================== ===================== =====================
    """

    DEFAULT = MappingProxyType(
        {
            "pfunc": MPFuncs.pf1,
            "npeaks": 5,
            "change_severity": None,
            "bfunc": None,
            "min_coord": 0.0,
            "max_coord": 100.0,
            "min_height": 30.0,
            "max_height": 70.0,
            "uniform_height": 50.0,
            "min_width": 0.0001,
            "max_width": 0.2,
            "uniform_width": 0.1,
            "lambda_": 0.0,
            "move_severity": 1.0,
            "height_severity": 7.0,
            "width_severity": 0.01,
            "period": 5000,
        }
    )

    ALT1 = MappingProxyType(
        {
            "pfunc": MPFuncs.pf2,
            "npeaks": 10,
            "change_severity": None,
            "bfunc": None,
            "min_coord": 0.0,
            "max_coord": 100.0,
            "min_height": 30.0,
            "max_height": 70.0,
            "uniform_height": 50.0,
            "min_width": 1.0,
            "max_width": 12.0,
            "uniform_width": 0,
            "lambda_": 0.5,
            "move_severity": 1.0,
            "height_severity": 7.0,
            "width_severity": 1.0,
            "period": 5000,
        }
    )

    ALT2 = MappingProxyType(
        {
            "pfunc": MPFuncs.pf2,
            "npeaks": 50,
            "change_severity": None,
            "bfunc": lambda x: 10,
            "min_coord": 0.0,
            "max_coord": 100.0,
            "min_height": 30.0,
            "max_height": 70.0,
            "uniform_height": 0,
            "min_width": 1.0,
            "max_width": 12.0,
            "uniform_width": 0,
            "lambda_": 0.5,
            "move_severity": 1.0,
            "height_severity": 1.0,
            "width_severity": 0.5,
            "period": 1000,
        }
    )
