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
from __future__ import annotations
from typing import Callable
from operator import gt, ge, le, lt, eq, ne


# ====================================================================================== #
class Node:
    def __init__(self, dimensions: int, cargo: tuple = None):
        self.cargo = cargo
        self.next = [None] * dimensions
        self.prev = [None] * dimensions
        self.ignore = 0
        self.area = [0.0] * dimensions
        self.volume = [0.0] * dimensions

    # -------------------------------------------------------- #
    def compare(self, other: Node, op: Callable) -> bool:
        if self.cargo is None or other.cargo is None:
            return False
        zipper = zip(self.cargo, other.cargo)
        true = [op(a, b) for a, b in zipper]
        return all(true)

    # -------------------------------------------------------- #
    def __gt__(self, other: Node) -> bool:
        return self.compare(other, gt)

    def __ge__(self, other: Node) -> bool:
        return self.compare(other, ge)

    def __le__(self, other: Node) -> bool:
        return self.compare(other, le)

    def __lt__(self, other: Node) -> bool:
        return self.compare(other, lt)

    def __eq__(self, other: Node) -> bool:
        return self.compare(other, eq)

    def __ne__(self, other: Node) -> bool:
        return self.compare(other, ne)

    # -------------------------------------------------------- #
    def __str__(self) -> str:
        return str(self.cargo)

    # -------------------------------------------------------- #
    def __hash__(self) -> int:
        return hash(self.cargo)
