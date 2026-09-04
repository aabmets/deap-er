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
from collections.abc import Callable
from functools import partial
from typing import Any

from .dtypes import Individual


class LintHints:
    __test__: Callable[..., Any]

    map: partial[Any]
    clone: partial[Any]

    attr_int: Callable[..., Any]
    attr_bool: Callable[..., Any]
    attr_float: Callable[..., Any]
    attr_item: Callable[..., Any]

    individual: Callable[..., Individual]
    individuals: Callable[..., list[Individual]]
    population: Callable[..., list[Individual]]
    populations: Callable[..., list[list[Individual]]]
    particle: Callable[..., Individual]
    particles: Callable[..., list[Individual]]
    swarm: Callable[..., list[Individual]]
    swarms: Callable[..., list[list[Individual]]]

    evaluate: Callable[..., Any]
    select: Callable[..., list[Individual]]
    mate: Callable[..., tuple[Individual, Individual]]
    mutate: Callable[..., tuple[Individual]]
    generate: Callable[..., list[Individual]]
    update: Callable[..., Any]
