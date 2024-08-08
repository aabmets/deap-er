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
from typing import Callable
from functools import partial


# ====================================================================================== #
class LintHints:
    __test__: Callable

    map: partial
    clone: partial

    attr_int: Callable
    attr_bool: Callable
    attr_float: Callable
    attr_item: Callable

    individual: Callable
    individuals: Callable
    population: Callable
    populations: Callable
    particle: Callable
    particles: Callable
    swarm: Callable
    swarms: Callable

    evaluate: Callable
    select: Callable
    mate: Callable
    mutate: Callable
    generate: Callable
    update: Callable
