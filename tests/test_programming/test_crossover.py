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
import operator
import os
import subprocess
import sys
from typing import Any, override

import numpy
from deap_er import gp, tools
from deap_er.private.programming.columnar import Array, Window


def _assert_well_typed(tree: gp.PrimitiveTree) -> None:
    stack: list[type] = []
    for node in reversed(list(tree)):
        if node.arity == 0:
            stack.append(node.ret)
            continue
        args = [stack.pop() for _ in range(node.arity)]
        assert args == list(node.args)
        stack.append(node.ret)


def test_typed_object_root_does_not_swap_incompatible_subtrees():
    def wrap_int(value: int) -> object:
        return value

    def wrap_str(value: str) -> object:
        return value

    def concat(left: str, right: str) -> str:
        return left + right

    pset = gp.PrimitiveSetTyped("main", [], object)
    pset.add_primitive(wrap_int, [int], object, name="wrap_int")
    pset.add_primitive(wrap_str, [str], object, name="wrap_str")
    pset.add_primitive(operator.add, [int, int], int, name="add")
    pset.add_primitive(concat, [str, str], str, name="cat")
    pset.add_terminal(1, int, name="one")
    pset.add_terminal("x", str, name="ex")

    for seed in range(40):
        tools.rng.seed(seed)
        first: Any = gp.PrimitiveTree.from_string("wrap_int(add(one, one))", pset)
        second: Any = gp.PrimitiveTree.from_string("wrap_str(cat(ex, ex))", pset)
        gp.cx_one_point(first, second)
        _assert_well_typed(first)
        _assert_well_typed(second)


class RankedType(type):
    """Metaclass whose hash is a fixed rank, so set order is known."""

    rank: int

    @override
    def __hash__(cls) -> int:
        return cls.rank


class A(metaclass=RankedType):
    rank = 1


class B(metaclass=RankedType):
    rank = 2


class C(metaclass=RankedType):
    rank = 0


PROGRAM_ORDER: list[type] = [A, B, C]


def _ranked_pset() -> gp.PrimitiveSetTyped:
    def three(first: Any, second: Any, third: Any) -> Any:
        return first

    pset = gp.PrimitiveSetTyped("ranked", [], A)
    pset.add_primitive(three, [A, B, C], A, name="abc")
    pset.add_primitive(three, [B, C, A], A, name="bca")
    for type_, name in ((A, "a"), (B, "b"), (C, "c")):
        pset.add_terminal(name, type_, name=f"{name}1")
        pset.add_terminal(name, type_, name=f"{name}2")
    return pset


def _ranked_parents(pset: gp.PrimitiveSetTyped) -> tuple[Any, Any]:
    first = gp.PrimitiveTree.from_string("abc(a1, b1, c1)", pset)
    second = gp.PrimitiveTree.from_string("bca(b2, c2, a2)", pset)
    return first, second


def _swapped_type(before: Any, after: Any) -> type:
    changed: list[type] = [
        new.ret for old, new in zip(before, after, strict=True) if old is not new
    ]
    assert len(changed) == 1
    return changed[0]


def test_set_order_differs_from_program_order():
    assert list({A, B, C}) != PROGRAM_ORDER


def test_cx_one_point_draws_type_in_program_order():
    pset = _ranked_pset()
    for seed in range(30):
        tools.rng.seed(seed)
        expected = tools.rng.choice(PROGRAM_ORDER)
        tools.rng.seed(seed)
        first, second = _ranked_parents(pset)
        before = list(first)
        gp.cx_one_point(first, second)
        assert _swapped_type(before, first) is expected


def test_cx_one_point_leaf_biased_draws_type_in_program_order():
    pset = _ranked_pset()
    for seed in range(30):
        tools.rng.seed(seed)
        tools.rng.random()
        tools.rng.random()
        expected = tools.rng.choice(PROGRAM_ORDER)
        tools.rng.seed(seed)
        first, second = _ranked_parents(pset)
        before = list(first)
        gp.cx_one_point_leaf_biased(first, second, 1.0)
        assert _swapped_type(before, first) is expected


def test_cx_homologous_fallback_draws_type_in_program_order():
    pset = _ranked_pset()
    for seed in range(30):
        tools.rng.seed(seed)
        tools.rng.integers(1, 4)
        expected = tools.rng.choice(PROGRAM_ORDER)
        tools.rng.seed(seed)
        first, second = _ranked_parents(pset)
        before = list(first)
        gp.cx_homologous(first, second)
        assert _swapped_type(before, first) is expected


def test_cx_one_point_semantic_draws_type_in_program_order():
    pset = gp.make_column_pset(["first", "second"])
    gp.add_numpy_primitives(pset)
    gp.add_window_primitives(pset)
    gp.add_window_ephemeral(pset, "CX_TYPE_ORDER", 2, 5)
    matrix = numpy.random.default_rng(3).normal(size=(12, 2))
    for seed in range(30):
        first: Any = gp.PrimitiveTree.from_string("rolling_sum(first, 3)", pset)
        second: Any = gp.PrimitiveTree.from_string("rolling_mean(second, 4)", pset)
        tools.rng.seed(seed)
        expected = tools.rng.choice([Array, Window])
        tools.rng.seed(seed)
        before = list(first)
        gp.cx_one_point_semantic(first, second, pset, matrix)
        assert _swapped_type(before, first) is expected


_HASHSEED_SCRIPT = """
from deap_er import gp, tools
from deap_er.private.programming.columnar import Array, Window

class NamedType(type):
    def __hash__(cls):
        return hash(cls.__name__)

A = NamedType("A", (), {})
B = NamedType("B", (), {})
C = NamedType("C", (), {})
D = NamedType("D", (), {})

def three(first, second, third, fourth):
    return first

pset = gp.PrimitiveSetTyped("named", [], A)
pset.add_primitive(three, [A, B, C, D], A, name="abcd")
pset.add_primitive(three, [D, C, B, A], A, name="dcba")
for type_, name in ((A, "a"), (B, "b"), (C, "c"), (D, "d")):
    pset.add_terminal(name, type_, name=name + "1")
    pset.add_terminal(name, type_, name=name + "2")

for operator in (gp.cx_one_point, gp.cx_homologous):
    for seed in range(20):
        tools.rng.seed(seed)
        first = gp.PrimitiveTree.from_string("abcd(a1, b1, c1, d1)", pset)
        second = gp.PrimitiveTree.from_string("dcba(d2, c2, b2, a2)", pset)
        operator(first, second)
        print(first, second)
"""


def test_seeded_typed_crossover_ignores_hash_seed():
    outputs = set()
    for hash_seed in ("1", "2", "3", "4", "5"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", _HASHSEED_SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1
