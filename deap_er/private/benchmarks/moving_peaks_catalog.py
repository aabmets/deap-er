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
import math
from collections.abc import Iterable, Sequence
from types import MappingProxyType
from typing import Any

__all__: list[str] = ["MPFuncs", "MPConfigs"]


class MPFuncs:
    """Peak functions for Moving Peaks custom presets."""

    @staticmethod
    def pf1(
        individual: Sequence[float], positions: Iterable[float], height: float, width: float
    ) -> float:
        """The peak function of the :data:`DEFAULT` preset.

        Official Moving Peaks scenario 1 is the squared form
        ``height / (1 + width * sum((x_i - p_i)^2))``.

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
            "move_severity": 1.5,
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
