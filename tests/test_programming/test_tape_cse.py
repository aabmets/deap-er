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
import weakref

import numpy
import pytest
from deap_er import gp, tools
from deap_er.private.programming import tape_cse
from deap_er.private.programming.opcodes import apply_opcode, interpret_tape

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
    original = apply_opcode

    def counted(stack, columns, tape, opcode, operand):
        if opcode == int(gp.Opcode.MUL):
            calls["count"] += 1
        return original(stack, columns, tape, opcode, operand)

    monkeypatch.setattr(tape_cse, "apply_opcode", counted)
    tape_cse.run_opcode_cse(tapes, matrix)
    assert calls["count"] == 1


def test_identical_full_tapes_share_one_root_node():
    tape = _tape("vadd(first, second)")
    plan = tape_cse.build_cse_plan([tape, tape, tape])
    assert len(plan.nodes) == 3
    assert plan.roots == (2, 2, 2)


def test_identical_roots_each_get_their_row():
    matrix = _matrix(_samples())
    tape = _tape("vadd(first, second)")
    actual = tape_cse.run_opcode_cse([tape, tape, tape], matrix)
    expected = numpy.asarray(gp.interpret_tape(tape, matrix), dtype=numpy.float64)
    for row in actual:
        numpy.testing.assert_allclose(row, expected, equal_nan=True)


def _track_live_columns(monkeypatch):
    """Count the node columns still referenced each time a node is evaluated."""
    produced: list[weakref.ref[numpy.ndarray]] = []
    peak = {"live": 0}

    def note_live():
        live = sum(ref() is not None for ref in produced)
        peak["live"] = max(peak["live"], live)

    def tracked_interpret(tape, columns):
        note_live()
        value = interpret_tape(tape, columns)
        if isinstance(value, numpy.ndarray):
            produced.append(weakref.ref(value))
        return value

    def tracked_apply(stack, columns, tape, opcode, operand):
        note_live()
        apply_opcode(stack, columns, tape, opcode, operand)
        produced.append(weakref.ref(stack[-1]))

    monkeypatch.setattr(tape_cse, "interpret_tape", tracked_interpret)
    monkeypatch.setattr(tape_cse, "apply_opcode", tracked_apply)
    return peak


def test_cse_frees_each_node_column_after_its_last_consumer(monkeypatch):
    matrix = _matrix(_samples())
    exprs = [
        f"rolling_mean(vadd(first, rolling_mean(second, {inner})), {outer})"
        for inner, outer in itertools.product(range(1, 5), range(1, 5))
    ]
    tapes = [_tape(expr) for expr in exprs]
    plan = tape_cse.build_cse_plan(tapes)
    assert len(plan.nodes) == 26
    expected = numpy.stack([numpy.asarray(gp.interpret_tape(tape, matrix)) for tape in tapes])
    peak = _track_live_columns(monkeypatch)
    actual = tape_cse.run_opcode_cse(tapes, matrix)
    numpy.testing.assert_array_equal(actual, expected)
    # Only the two leaves, the vadd shared by four tapes and the
    # current tape's stack may be live; keeping every node gives 25.
    assert peak["live"] <= 4


def _constant_kit():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    gp.add_window_primitives(pset)
    pset.add_terminal(2.0, gp.Array, "two")
    pset.add_terminal(5.0, gp.Array, "five")
    return pset


def test_combined_node_tapes_keep_each_child_constant_pool():
    pset = _constant_kit()
    shared = gp.lower_tree("vmul(first, two)", pset)
    reordered = gp.lower_tree("vadd(five, vmul(first, two))", pset)
    plan = tape_cse.build_cse_plan([shared, reordered])
    matrix = _matrix(_samples())
    root = plan.nodes[plan.roots[1]]
    numpy.testing.assert_allclose(
        gp.interpret_tape(root.tape, matrix), gp.interpret_tape(reordered, matrix)
    )


@pytest.mark.parametrize(
    "expr",
    ["rolling_mean(two, 3)", "diff(five, 2)", "ts_rank(two, 3)", "vadd(first, ema(two, 4))"],
)
def test_a_window_over_a_constant_reads_it_as_a_constant_column(expr):
    pset = _constant_kit()
    gp.add_ts_primitives(pset)
    matrix = _matrix(_samples())
    tape = gp.lower_tree(expr, pset)
    column = numpy.full(matrix.shape[0], 2.0 if "two" in expr else 5.0)
    as_column = gp.lower_tree(expr.replace("two", "second").replace("five", "second"), pset)
    expected = gp.interpret_tape(as_column, numpy.column_stack([matrix[:, 0], column, column]))
    numpy.testing.assert_allclose(gp.interpret_tape(tape, matrix), expected, equal_nan=True)
    numpy.testing.assert_allclose(gp.interpret_tapes([tape], matrix)[0], expected, equal_nan=True)
