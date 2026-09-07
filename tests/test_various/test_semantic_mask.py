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
from deap_er import gp, tools
from deap_er.private.various.semantic_mask import as_semantic_matrix

_LEAKED = (
    "as_semantic_matrix",
    "packed_semantics",
    "validate_semantic_matrix",
    "semantic_column_keep",
)


def test_as_semantic_matrix_docstring_does_not_claim_c_contiguous():
    doc = as_semantic_matrix.__doc__
    assert doc is not None
    assert "C-contiguous" not in doc


def test_as_semantic_matrix_keeps_fortran_order():
    fortran = numpy.asfortranarray([[1.0, 2.0], [3.0, 4.0]])
    packed = as_semantic_matrix(fortran)
    assert packed.dtype == numpy.float64
    assert packed.ndim == 2
    assert not packed.flags["C_CONTIGUOUS"]
    numpy.testing.assert_array_equal(packed, fortran)


def test_tools_and_gp_do_not_export_semantic_mask_helpers():
    for name in _LEAKED:
        assert not hasattr(tools, name)
        assert not hasattr(gp, name)
    assert hasattr(tools, "semantic_valid_mask")
