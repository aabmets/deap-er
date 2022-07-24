# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 - Mattias Aabmets, The DEAP Team and Other Contributors           #
#                                                                                        #
#   Permission is hereby granted, free of charge, to any person obtaining a copy         #
#   of this software and associated documentation files (the "Software"), to deal        #
#   in the Software without restriction, including without limitation the rights         #
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell            #
#   copies of the Software, and to permit persons to whom the Software is                #
#   furnished to do so, subject to the following conditions:                             #
#                                                                                        #
#   The above copyright notice and this permission notice shall be included in all       #
#   copies or substantial portions of the Software.                                      #
#                                                                                        #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR           #
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,             #
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE          #
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER               #
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,        #
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE        #
#   SOFTWARE.                                                                            #
#                                                                                        #
# ====================================================================================== #
from types import MappingProxyType
from deap_er.base import Individual
from typing import Iterable, Optional
import itertools
import random
import math


# ====================================================================================== #
class MovingPeaks:
    """
    | The Moving Peaks Benchmark is a fitness function changing over time.
    | It consists of a number of peaks changing in height, width and location.
    | The default configuration of the Moving Peaks benchmark is :data:`PeaksConfig.DEFAULT`.

    :param dimensions: The dimensionality of the search domain.
    :param kwargs: Keyword arguments, optional.

    .. dropdown:: Table of Kwargs
       :margin: 0 5 0 0

       =================== ========== =================================================================================
       Parameter           Type       Details
       =================== ========== =================================================================================
       ``pfunc``           *Callable* The peak function or a list of peak functions.
       ``bfunc``           *Callable* Basis function for static landscape.
       ``npeaks``          *NumOrSeq* Number of peaks. An integer or a sequence of three integers [min, initial, max].
       ``min_coord``       *float*    Minimum coordinate for the centre of the peaks.
       ``max_coord``       *float*    Maximum coordinate for the centre of the peaks.
       ``min_height``      *float*    Minimum height of the peaks.
       ``max_height``      *float*    Maximum height of the peaks.
       ``uniform_height``  *float*    Starting height of all peaks. Random, if ``uniform_height <= 0``.
       ``min_width``       *float*    Minimum width of the peaks.
       ``max_width``       *float*    Maximum width of the peaks
       ``uniform_width``   *float*    Starting width of all peaks. Random, if ``uniform_width <= 0``.
       ``lambda``          *float*    Correlation between changes.
       ``move_severity``   *float*    The distance a single peak moves when peaks change.
       ``height_severity`` *float*    The standard deviation of the change to the height of a peak when peaks change.
       ``width_severity``  *float*    The standard deviation of the change to the width of a peak when peaks change.
       ``period``          *int*      Period between two changes.
       =================== ========== =================================================================================
    """
    def __init__(self, dimensions: int, **kwargs: Optional):
        self.dim = dimensions
        sc = PeakConfigs.DEFAULT.copy()  # default config
        sc.update(kwargs)

        n_peaks = sc.get("npeaks")
        pfunc = sc.get("pfunc")

        # ------------------------------------ #
        self.min_peaks, self.max_peaks = None, None
        if hasattr(n_peaks, "__getitem__"):
            self.min_peaks, n_peaks, self.max_peaks = n_peaks
            self.number_severity = sc.get("number_severity")
        try:
            if len(pfunc) == n_peaks:
                self.peaks_function = pfunc
            else:
                self.peaks_function = random.sample(pfunc, n_peaks)
            self.pfunc_pool = tuple(pfunc)
        except TypeError:
            self.peaks_function = list(itertools.repeat(pfunc, n_peaks))
            self.pfunc_pool = (pfunc,)

        # ------------------------------------ #
        self.last_change_vector = [
            [
                random.random() - 0.5
                for _ in range(dimensions)
            ] for _ in range(n_peaks)
        ]
        # ------------------------------------ #
        self.min_coord = sc.get("min_coord")
        self.max_coord = sc.get("max_coord")
        self.peaks_position = [
            [
                random.uniform(self.min_coord, self.max_coord)
                for _ in range(dimensions)
            ] for _ in range(n_peaks)
        ]
        # ------------------------------------ #
        uniform_height = sc.get("uniform_height")
        self.min_height = sc.get("min_height")
        self.max_height = sc.get("max_height")
        if uniform_height != 0:
            self.peaks_height = [uniform_height for _ in range(n_peaks)]
        else:
            def rand_height():
                return random.uniform(self.min_height, self.max_height)
            self.peaks_height = [rand_height() for _ in range(n_peaks)]

        # ------------------------------------ #
        uniform_width = sc.get("uniform_width")
        self.min_width = sc.get("min_width")
        self.max_width = sc.get("max_width")
        if uniform_width != 0:
            self.peaks_width = [uniform_width for _ in range(n_peaks)]
        else:
            def rand_width():
                return random.uniform(self.min_width, self.max_width)
            self.peaks_width = [rand_width() for _ in range(n_peaks)]

        # ------------------------------------ #
        self.basis_function = sc.get("bfunc")
        self.move_severity = sc.get("move_severity")
        self.height_severity = sc.get("height_severity")
        self.width_severity = sc.get("width_severity")
        self.period = sc.get("period")
        self.lamb = sc.get("lambda")
        self._optimum = None
        self._error = None
        self._offline_error = 0
        self.nevals = 0

    # -------------------------------------------------------- #
    @property
    def global_maximum(self) -> tuple:
        """
        Returns the value and position of the largest peak.
        """
        potential_max = list()
        zipper = zip(
            self.peaks_function, self.peaks_position,
            self.peaks_height, self.peaks_width
        )
        for func, pos, height, width in zipper:
            result = func(pos, pos, height, width)
            value: tuple = (result, pos)
            potential_max.append(value)
        return max(potential_max)

    # -------------------------------------------------------- #
    @property
    def sorted_maxima(self) -> list:
        """
        Returns all visible peak values and positions, sorted from the largest to the smallest peaks.
        """
        maximums = list()
        zipper = zip(
            self.peaks_function, self.peaks_position,
            self.peaks_height,  self.peaks_width
        )
        for func, pos, height, width in zipper:
            result = func(pos, pos, height, width)
            if result >= self.__call__(pos, count=False):
                value: tuple = (result, pos)
                maximums.append(value)
        return sorted(maximums, reverse=True)

    # -------------------------------------------------------- #
    @property
    def offline_error(self) -> float:
        """
        Returns the offline error of the landscape.
        """
        return self._offline_error / self.nevals

    # -------------------------------------------------------- #
    @property
    def current_error(self) -> Optional[float]:
        """
        Returns the current error of the landscape.
        """
        return self._error

    # -------------------------------------------------------- #
    def __call__(self, individual: Individual, count: bool = True) -> float:
        """
        Evaluate the given **individual** in the context of the current configuration.

        :param individual: The individual to be evaluated.
        :param count: Whether to count this evaluation in the
            total evaluation count, optional.
        :return: The fitness of the individual.
        :type individual: :ref:`Individual <datatypes>`
        """
        possible_values = []
        zipper = zip(
            self.peaks_function, self.peaks_position,
            self.peaks_height, self.peaks_width
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
            if self._optimum is None:
                self._optimum = self.global_maximum[0]
                self._error = abs(fitness - self._optimum)
            self._error = min(self._error, abs(fitness - self._optimum))
            self._offline_error += self._error

            if self.period > 0 and self.nevals % self.period == 0:
                self.change_peaks()

        return fitness

    # -------------------------------------------------------- #
    def change_peaks(self):
        """
        Changes the position, the height, the width and the number of peaks.
        """
        self._optimum = None

        if self.min_peaks is not None and self.max_peaks is not None:
            n_peaks = len(self.peaks_function)
            u = random.random()
            r = self.max_peaks - self.min_peaks
            if u < 0.5:
                u = random.random()
                runs = int(round(r * u * self.number_severity))
                n = min(n_peaks - self.min_peaks, runs)
                for i in range(n):
                    len_ = len(self.peaks_function)
                    idx = random.randrange(len_)
                    self.peaks_function.pop(idx)
                    self.peaks_position.pop(idx)
                    self.peaks_height.pop(idx)
                    self.peaks_width.pop(idx)
                    self.last_change_vector.pop(idx)
            else:
                u = random.random()
                runs = int(round(r * u * self.number_severity))
                n = min(self.max_peaks - n_peaks, runs)
                for i in range(n):
                    rand = random.choice(self.pfunc_pool)
                    self.peaks_function.append(rand)
                    rand = [
                        random.uniform(self.min_coord, self.max_coord)
                        for _ in range(self.dim)
                    ]
                    self.peaks_position.append(rand)
                    rand = random.uniform(self.min_height, self.max_height)
                    self.peaks_height.append(rand)
                    rand = random.uniform(self.min_width, self.max_width)
                    self.peaks_width.append(rand)
                    rand = [random.random() - 0.5 for _ in range(self.dim)]
                    self.last_change_vector.append(rand)

        for i in range(len(self.peaks_function)):
            def fn_shift(s, c):
                return shift_length * (1.0 - self.lamb) * s + self.lamb * c

            len_ = len(self.peaks_position[i])
            shift = [random.random() - 0.5 for _ in range(len_)]
            shift_length = sum(s ** 2 for s in shift)
            if shift_length > 0:
                shift_length = self.move_severity / math.sqrt(shift_length)
            else:
                shift_length = 0

            zipper = zip(shift, self.last_change_vector[i])
            shift = [fn_shift(s, c) for s, c in zipper]
            shift_length = sum(s ** 2 for s in shift)
            if shift_length > 0:
                shift_length = self.move_severity / math.sqrt(shift_length)
            else:
                shift_length = 0

            shift = [s * shift_length for s in shift]

            new_position = []
            final_shift = []
            for pp, s in zip(self.peaks_position[i], shift):
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

            self.peaks_position[i] = new_position
            self.last_change_vector[i] = final_shift

            def change_shape(axis, axis_min, axis_max, sev):
                change = random.gauss(0, 1) * sev
                new_value = change + axis[i]
                if new_value < axis_min:
                    axis[i] = 2.0 * axis_min - axis[i] - change
                elif new_value > axis_max:
                    axis[i] = 2.0 * axis_max - axis[i] - change
                else:
                    axis[i] = new_value

            change_shape(
                self.peaks_height,
                self.min_height,
                self.max_height,
                self.height_severity
            )
            change_shape(
                self.peaks_width,
                self.min_width,
                self.max_width,
                self.width_severity
            )


# ====================================================================================== #
class PeakFuncs:
    """
    | This class contains the peak functions for the Moving Peaks problem.
    | These functions can be used for creating custom configuration presets.
    """
    @staticmethod
    def pf1(individual: Individual, positions: Iterable,
            height: float, width: float) -> float:
        """
        The peak function of the :data:`DEFAULT` preset.

        :param individual: The individual to be evaluated.
        :param positions: The positions of the peaks.
        :param height: The height of the peaks.
        :param width: The width of the peaks.
        :return: The fitness of the individual.
        """
        value = 0.0
        for x, p in zip(individual, positions):
            value += (x - p) ** 2
        return height / (1 + width * value)

    # -------------------------------------------------------- #
    @staticmethod
    def pf2(individual: Individual, positions: Iterable,
            height: float, width: float) -> float:
        """
        The peak function of the :data:`ALT1` and :data:`ALT2` presets.

        :param individual: The individual to be evaluated.
        :param positions: The positions of the peaks.
        :param height: The height of the peaks.
        :param width: The width of the peaks.
        :return: The fitness of the individual.
        """
        value = 0.0
        for x, p in zip(individual, positions):
            value += (x - p) ** 2
        return height - width * math.sqrt(value)

    # -------------------------------------------------------- #
    @staticmethod
    def pf3(individual: Individual, positions: Iterable,
            height: float, *_) -> float:
        """
        An optional peak function.

        :param individual: The individual to be evaluated.
        :param positions: The positions of the peaks.
        :param height: The height of the peaks.
        :return: The fitness of the individual.
        """
        value = 0.0
        for x, p in zip(individual, positions):
            value += (x - p) ** 2
        return height * value


# ====================================================================================== #
class PeakConfigs:
    """
    | This class contains the configuration presets for the Moving Peaks problem.
    | The presets are of type :data:`dict` and can be accessed as **class attributes**.

    .. dropdown:: Table of Presets
       :margin: 0 5 0 0

        =================== ===================== ===================== =====================
        Keys / Presets      **DEFAULT**           **ALT1**              **ALT2**
        =================== ===================== ===================== =====================
        ``pfunc``           :func:`PeakFuncs.pf1` :func:`PeakFuncs.pf2` :func:`PeakFuncs.pf2`
        ``bfunc``           :obj:`None`           :obj:`None`           :obj:`lambda x: 10`
        ``npeaks``          5                     10                    50
        ``min_coord``       0.0                   0.0                   0.0
        ``max_coord``       100.0                 100.0                 100.0
        ``min_height``      30.0                  30.0                  30.0
        ``max_height``      70.0                  70.0                  70.0
        ``uniform_height``  50.0                  50.0                  0.0
        ``min_width``       0.0001                1.0                   1.0
        ``max_width``       0.2                   12.0                  12.0
        ``uniform_width``   0.1                   0.0                   0.0
        ``lambda``          0.0                   0.5                   0.5
        ``move_severity``   1.0                   1.5                   1.0
        ``height_severity`` 7.0                   7.0                   1.0
        ``width_severity``  0.01                  1.0                   0.5
        ``period``          5000                  5000                  1000
        =================== ===================== ===================== =====================
    """
    # -------------------------------------------------------- #
    DEFAULT = MappingProxyType(
        {
            "pfunc": PeakFuncs.pf1,
            "npeaks": 5,
            "bfunc": None,
            "min_coord": 0.0,
            "max_coord": 100.0,
            "min_height": 30.0,
            "max_height": 70.0,
            "uniform_height": 50.0,
            "min_width": 0.0001,
            "max_width": 0.2,
            "uniform_width": 0.1,
            "lambda": 0.0,
            "move_severity": 1.0,
            "height_severity": 7.0,
            "width_severity": 0.01,
            "period": 5000
        }
    )

    # -------------------------------------------------------- #
    ALT1 = MappingProxyType(
        {
            "pfunc": PeakFuncs.pf2,
            "npeaks": 10,
            "bfunc": None,
            "min_coord": 0.0,
            "max_coord": 100.0,
            "min_height": 30.0,
            "max_height": 70.0,
            "uniform_height": 50.0,
            "min_width": 1.0,
            "max_width": 12.0,
            "uniform_width": 0,
            "lambda": 0.5,
            "move_severity": 1.0,
            "height_severity": 7.0,
            "width_severity": 1.0,
            "period": 5000
        }
    )

    # -------------------------------------------------------- #
    ALT2 = MappingProxyType(
        {
            "pfunc": PeakFuncs.pf2,
            "npeaks": 50,
            "bfunc": lambda x: 10,
            "min_coord": 0.0,
            "max_coord": 100.0,
            "min_height": 30.0,
            "max_height": 70.0,
            "uniform_height": 0,
            "min_width": 1.0,
            "max_width": 12.0,
            "uniform_width": 0,
            "lambda": 0.5,
            "move_severity": 1.0,
            "height_severity": 1.0,
            "width_severity": 0.5,
            "period": 1000
        }
    )


# def diversity(population):
#     nind = len(population)
#     ndim = len(population[0])
#     d = [0.0] * ndim
#     for x in population:
#         d = [di + xi for di, xi in zip(d, x)]
#     d = [di / nind for di in d]
#     values = [
#         (di - xi) ** 2
#         for x in population
#         for di, xi in zip(d, x)
#     ]
#     return math.sqrt(sum(values))
#
#
# if __name__ == "__main__":
#     mpb = MovingPeaks(dim=2, npeaks=[1, 1, 10], number_severity=0.1)
#     print(mpb.maxima())
#     mpb.change_peaks()
#     print(mpb.maxima())
