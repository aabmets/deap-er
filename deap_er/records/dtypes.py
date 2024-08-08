#
#   MIT License
#   
#   Copyright (c) 2022, Mattias Aabmets
#   
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#   
#   SPDX-License-Identifier: MIT
#
from .hall_of_fame import *
from .statistics import *
from .logbook import *
from typing import Union, Tuple


__all__ = ['Hof', 'Stats', 'AlgoResult']


Hof = Union[HallOfFame, ParetoFront]
""":meta private:"""

Stats = Union[Statistics, MultiStatistics]
""":meta private:"""

AlgoResult = Tuple[list, Logbook]
""":meta private:"""
