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
from deap_er import Fitness, creator, gp, tools

FIT = "SEM_ARC_FIT"
IND = "SEM_ARC_IND"


def _trees():
    pset = gp.make_column_pset(["first"])
    gp.add_numpy_primitives(pset)
    pset.add_terminal(2.5, gp.Array, "two_half")
    constant = gp.PrimitiveTree([pset.mapping["two_half"]])
    column = gp.PrimitiveTree([pset.mapping["first"]])
    return pset, constant, column


def test_interpret_tapes_moments_fill_archives():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, gp.PrimitiveTree, fitness=creator.__dict__[FIT])
    try:
        pset, constant, column = _trees()
        first = creator.__dict__[IND](constant)
        second = creator.__dict__[IND](column)
        first.fitness.values = (0.4,)
        second.fitness.values = (0.8,)
        before = (first.fitness.values, second.fitness.values)
        series = numpy.linspace(0.0, 3.0, 8)
        matrix = numpy.ascontiguousarray(series.reshape(8, 1))
        tapes = [gp.lower_tree(first, pset), gp.lower_tree(second, pset)]
        predicted = gp.interpret_tapes(tapes, matrix, backend="opcode")
        descriptors = tools.semantic_moments(predicted)
        lows = descriptors.min(axis=0) - 1.0
        highs = descriptors.max(axis=0) + 1.0
        grid = tools.GridArchive(ranges=list(zip(lows, highs, strict=True)), bins=4)
        cvt = tools.CvtArchive(descriptors)
        unstructured = tools.UnstructuredArchive(4, min_distance=0.05)
        assert grid.add(first, descriptors[0])
        assert grid.add(second, descriptors[1])
        assert cvt.add(first, descriptors[0])
        assert unstructured.add(second, descriptors[1])
        assert len(grid) == 2
        assert grid.descriptor_to_index(descriptors[0]) != grid.descriptor_to_index(descriptors[1])
        assert first.fitness.values == before[0]
        assert second.fitness.values == before[1]
        assert not numpy.array_equal(predicted[0], predicted[1])
    finally:
        del creator.__dict__[FIT]
        del creator.__dict__[IND]
