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
from deap_er import tools


def test_sorting_network_sorts_and_reports_errors():
    network = tools.SortingNetwork(4, [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)])

    values = [3, 1, 4, 2]
    network.sort(values)
    assert values == [1, 2, 3, 4]
    assert network.evaluate() == 0
    assert network.evaluate([[1, 0, 0, 0]]) == 0
    assert network.depth >= 1
    assert network.length >= 1
    assert len(network) == network.depth
    assert network[0] in network


def test_sorting_network_ignores_same_wire_and_packs_levels():
    network = tools.SortingNetwork(3)
    assert network.depth == 0
    network.add_connector(1, 1)
    assert network.depth == 0
    network.add_connector(0, 1)
    network.add_connector(0, 2)
    network.add_connector(1, 2)
    network[0] = list(network[0])
    assert any((0, 1) in level for level in network)
    assert list(network) == network.data
    if len(network) > 1:
        del network[-1]
    diagram = tools.SortingNetwork(3, [(0, 1), (0, 2), (1, 2)]).draw()
    assert "o" in diagram
    assert "0" in diagram
