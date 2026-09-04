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
from .hall_of_fame import *
from .statistics import *
from .logbook import *
from deap_er.base.dtypes import Individual


__all__ = ["Hof", "Stats", "AlgoResult", "Individual"]


type Hof = HallOfFame | ParetoFront
""":meta private:"""

type Stats = Statistics | MultiStatistics
""":meta private:"""

type AlgoResult = tuple[list[Individual], Logbook]
""":meta private:"""
