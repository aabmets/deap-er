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
from typing import Any

from deap_er.gp.primitives import PrimitiveSet, PrimitiveTree
from deap_er.gp.tools import build_tree_graph, compile_adf_tree, compile_tree, static_limit


def test_compile_tree_without_arguments():
    pset = PrimitiveSet("main", 0)
    pset.add_terminal(2, name="two")
    pset.add_terminal(3, name="three")
    pset.add_primitive(operator.add, 2)
    tree = PrimitiveTree.from_string("add(two, three)", pset)

    assert compile_tree(tree, pset) == 5


def test_compile_tree_sees_in_place_node_replacement():
    # A cache keyed only by id(tree) would keep the add() callable after
    # the root is replaced. Recompilation must read the current nodes.
    pset = PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.mul, 2)
    tree = PrimitiveTree.from_string("add(ARG0, 2)", pset)

    assert compile_tree(tree, pset)(3) == 5

    tree[0] = pset.mapping["mul"]

    assert compile_tree(tree, pset)(3) == 6


def test_compile_adf_tree_wires_automatically_defined_functions():
    adf = PrimitiveSet("ADF0", 2)
    adf.add_primitive(operator.add, 2)
    main = PrimitiveSet("MAIN", 1)
    main.add_adf(adf)
    main_tree = PrimitiveTree.from_string("ADF0(ARG0, ARG0)", main)
    adf_tree = PrimitiveTree.from_string("add(ARG0, ARG1)", adf)

    expressions: Any = [main_tree, adf_tree]
    func = compile_adf_tree(expressions, [main, adf])

    assert func(3) == 6


def test_build_tree_graph_returns_nodes_edges_and_labels():
    pset = PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_terminal(1, name="one")
    tree = PrimitiveTree.from_string("add(ARG0, one)", pset)

    nodes, edges, labels = build_tree_graph(tree)

    assert nodes == [0, 1, 2]
    assert (0, 1) in edges
    assert labels[0] == "add"
    assert 1 in labels


def test_static_limit_replaces_oversized_offspring():
    def grow(individual: list[int]) -> tuple[list[int]]:
        return (individual + [9],)

    limited = static_limit(len, 3)(grow)

    assert limited([1, 2]) == [[1, 2, 9]]
    assert limited([1, 2, 3, 4]) == [[1, 2, 3, 4]]
