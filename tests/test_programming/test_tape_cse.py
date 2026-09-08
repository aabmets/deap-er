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
import numpy
import pytest
from deap_er import gp, tools
from deap_er.private.programming import tape_cse
from deap_er.private.programming.opcodes import _apply_opcode

COLUMNS = ["first", "second", "third"]


def _kit(window_name="CSE_TEST", fill=None):
    pset = gp.make_column_pset(COLUMNS)
    kwargs = {} if fill is None else {"fill": fill}
    gp.add_numpy_primitives(pset, **kwargs)
    gp.add_window_primitives(pset)
    gp.add_window_ephemeral(pset, window_name, 1, 5)
    return pset


def _samples(size=24, seed=17):
    generator = numpy.random.default_rng(seed)
    columns = [
        generator.normal(size=size),
        generator.normal(size=size) * 2.0,
        numpy.abs(generator.normal(size=size)),
    ]
    columns[1][3] = 0.0
    columns[2][7] = numpy.nan
    return tuple(columns)


def _matrix(columns):
    return numpy.ascontiguousarray(numpy.stack(columns, axis=1))


def _lower_trees(pset, count, seed=23):
    tools.rng.seed(seed)
    tapes = []
    while len(tapes) < count:
        tree = gp.PrimitiveTree(gp.gen_half_and_half(pset, 2, 4))
        try:
            tapes.append(gp.lower_tree(tree, pset))
        except ValueError:
            continue
    return tapes


def _tape(expr):
    pset = _kit()
    return gp.lower_tree(gp.PrimitiveTree.from_string(expr, pset), pset)


def _tape_add_shared_mul(column: str):
    pset = _kit("CSE_SHARED")
    mapping = pset.mapping
    tree = gp.PrimitiveTree(
        [
            mapping["vadd"],
            mapping["vmul"],
            mapping["first"],
            mapping["second"],
            mapping[column],
        ]
    )
    return gp.lower_tree(tree, pset)


def test_cse_matches_naive_interpret_tape_on_random_trees():
    pset = _kit("CSE_RANDOM")
    columns = _samples()
    matrix = _matrix(columns)
    tapes = _lower_trees(pset, 12)
    actual = tape_cse.run_opcode_cse(tapes, matrix)
    expected = numpy.empty((len(tapes), matrix.shape[0]), dtype=numpy.float64)
    for index, tape in enumerate(tapes):
        expected[index] = numpy.asarray(gp.interpret_tape(tape, matrix), dtype=numpy.float64)
    numpy.testing.assert_allclose(actual, expected, equal_nan=True)


def test_interpret_tapes_cse_matches_interpret_tape_oracle():
    pset = _kit("CSE_BATCH")
    columns = _samples()
    matrix = _matrix(columns)
    tapes = _lower_trees(pset, 12)
    actual = gp.interpret_tapes(tapes, matrix)
    for index, tape in enumerate(tapes):
        expected = numpy.asarray(gp.interpret_tape(tape, matrix), dtype=numpy.float64)
        numpy.testing.assert_allclose(actual[index], expected, equal_nan=True)


@pytest.mark.parametrize(
    "expr",
    [
        "vadd(first, second)",
        "rolling_mean(first, 3)",
        "delay(first, 4)",
        "ema(first, 5)",
        "delay(rolling_mean(first, 5), 3)",
    ],
)
def test_cse_matches_oracle_on_windowed_expressions(expr):
    matrix = _matrix(_samples())
    tape = _tape(expr)
    actual = tape_cse.run_opcode_cse([tape], matrix)
    expected = numpy.asarray(gp.interpret_tape(tape, matrix), dtype=numpy.float64)
    numpy.testing.assert_allclose(actual[0], expected, equal_nan=True)


def test_shared_postfix_suffix_is_hash_consed_once():
    tapes = [_tape_add_shared_mul(column) for column in COLUMNS]
    plan = tape_cse.build_cse_plan(tapes)
    mul_keys = [
        node.key
        for node in plan.nodes
        if node.children and int(node.tape.opcodes[-1]) == int(gp.Opcode.MUL)
    ]
    assert len(mul_keys) == 1
    assert len(plan.nodes) < sum(tape.opcodes.size for tape in tapes)


def test_shared_postfix_suffix_applies_mul_once(monkeypatch):
    matrix = _matrix(_samples())
    tapes = [_tape_add_shared_mul(column) for column in COLUMNS]
    calls = {"count": 0}
    original = _apply_opcode

    def counted(stack, columns, tape, opcode, operand):
        if opcode == int(gp.Opcode.MUL):
            calls["count"] += 1
        return original(stack, columns, tape, opcode, operand)

    monkeypatch.setattr(tape_cse, "_apply_opcode", counted)
    tape_cse.run_opcode_cse(tapes, matrix)
    assert calls["count"] == 1


def test_identical_full_tapes_share_one_root_node():
    tape = _tape("vadd(first, second)")
    plan = tape_cse.build_cse_plan([tape, tape, tape])
    assert len(plan.nodes) == 3
    assert plan.roots == (2, 2, 2)


def test_evaluate_nodes_opcode_returns_one_slot_per_node_id():
    matrix = _matrix(_samples())
    plan = tape_cse.build_cse_plan([_tape_add_shared_mul(column) for column in COLUMNS])
    values = tape_cse._evaluate_nodes_opcode(plan.nodes, matrix)
    assert len(values) == len(plan.nodes)
    for node_id in range(len(plan.nodes)):
        assert isinstance(values[node_id], numpy.ndarray)
        assert values[node_id].shape == (matrix.shape[0],)


def test_compacting_evaluated_values_breaks_root_indexing():
    matrix = _matrix(_samples())
    plan = tape_cse.build_cse_plan([_tape("vadd(first, second)")])
    root = plan.roots[0]
    assert root == len(plan.nodes) - 1
    slots: list[numpy.ndarray | None] = [None] * len(plan.nodes)
    for node_id in range(len(plan.nodes)):
        slots[node_id] = numpy.full(matrix.shape[0], float(node_id))
    compacted = [value for value in slots if value is not None]
    assert len(compacted) == len(slots)
    slots[0] = None
    compacted = [value for value in slots if value is not None]
    assert len(compacted) < len(slots)
    with pytest.raises(IndexError):
        compacted[root]


def test_run_opcode_cse_requires_uncompacted_node_values(monkeypatch):
    matrix = _matrix(_samples())
    tapes = [_tape_add_shared_mul("third")]
    plan = tape_cse.build_cse_plan(tapes)
    original = tape_cse._evaluate_nodes_opcode

    def legacy_compacting(nodes, matrix):
        values = original(nodes, matrix)
        return values[1:]

    monkeypatch.setattr(tape_cse, "_evaluate_nodes_opcode", legacy_compacting)
    with pytest.raises(IndexError):
        tape_cse.run_opcode_cse(tapes, matrix)
    assert plan.roots[0] == len(plan.nodes) - 1
