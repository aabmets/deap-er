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
from typing import Any

import numpy
import pytest
from deap_er import tools


def test_mut_iso_line_float_moves_toward_donor(monkeypatch):
    monkeypatch.setattr(
        "deap_er.private.operators.mut_iso_line._sample_t",
        lambda iso: 0.5,
    )
    parent: Any = [0.0, 0.0]
    donor: Any = [1.0, 1.0]
    (mutant,) = tools.mut_iso_line(parent, donor, iso=0.0, sigma=0.0)
    assert mutant[0] == pytest.approx(0.5)
    assert mutant[1] == pytest.approx(0.5)


def test_mut_iso_line_clamps_to_bounds():
    parent: Any = [0.0]
    donor: Any = [10.0]
    tools.rng.seed(2)
    (mutant,) = tools.mut_iso_line(parent, donor, iso=0.0, sigma=0.0, low=0.0, up=1.0)
    assert 0.0 <= mutant[0] <= 1.0


def test_mut_iso_line_rejects_short_donor():
    parent: Any = [0.0, 0.0]
    donor: Any = [0.0]
    with pytest.raises(ValueError, match="Donor must be at least"):
        tools.mut_iso_line(parent, donor, iso=0.1, sigma=0.1)


def test_mut_iso_line_rejects_mismatched_bounds():
    parent: Any = [0.0]
    donor: Any = [1.0]
    with pytest.raises(ValueError, match="both be set or both omitted"):
        tools.mut_iso_line(parent, donor, iso=0.1, sigma=0.1, low=0.0)


def test_mut_iso_line_int_rounds_and_clamps():
    parent: Any = [1]
    donor: Any = [8]
    tools.rng.seed(3)
    (mutant,) = tools.mut_iso_line(parent, donor, iso=0.0, sigma=0.0, low=1, up=8)
    assert isinstance(mutant[0], int)
    assert 1 <= mutant[0] <= 8


def test_mut_iso_line_numpy_integer_gene_uses_int_path():
    parent: Any = [5]
    donor: Any = [8]
    tools.rng.seed(42)
    (py_mut,) = tools.mut_iso_line(parent, donor, iso=0.0, sigma=0.0, low=1, up=8)
    tools.rng.seed(42)
    np_parent: Any = [numpy.int64(5)]
    np_donor: Any = [numpy.int64(8)]
    (np_mut,) = tools.mut_iso_line(np_parent, np_donor, iso=0.0, sigma=0.0, low=1, up=8)
    assert isinstance(py_mut[0], int)
    assert isinstance(np_mut[0], int)
    assert np_mut[0] == py_mut[0]


def test_mut_iso_line_bit_can_copy_donor():
    parent: Any = [False]
    donor: Any = [True]
    tools.rng.seed(4)
    (mutant,) = tools.mut_iso_line(parent, donor, iso=0.0, sigma=0.0)
    assert mutant[0] is True


def test_mut_iso_line_preserves_numpy_bool_gene_type(monkeypatch):
    monkeypatch.setattr(
        "deap_er.private.operators.mut_iso_line._sample_t",
        lambda iso: 1.0,
    )
    parent: Any = [numpy.bool_(False)]
    donor: Any = [numpy.bool_(True)]
    (mutant,) = tools.mut_iso_line(parent, donor, iso=0.0, sigma=0.0)
    assert type(mutant[0]) is numpy.bool_
    assert mutant[0] == numpy.bool_(True)


def test_mut_heterogeneous_can_compose_iso_line_helpers(monkeypatch):
    monkeypatch.setattr(
        "deap_er.private.operators.mut_iso_line._sample_t",
        lambda iso: 1.0,
    )
    parent: Any = [0, 1, 0.0]
    donor: Any = [1, 8, 1.0]
    mutators = (
        lambda gene: tools.iso_line_bit(gene, donor[0], iso=0.0, sigma=0.0),
        lambda gene: tools.iso_line_int(gene, donor[1], iso=0.0, sigma=0.0, low=1, up=8),
        lambda gene: tools.iso_line_float(gene, donor[2], iso=0.0, sigma=0.0, low=0.0, up=1.0),
    )
    (mutant,) = tools.mut_heterogeneous(parent, mutators, 1.0)
    assert mutant == [True, 8, 1.0]


def test_mut_iso_line_zero_noise_is_deterministic_with_seed():
    parent: Any = [0.0, 2.0]
    donor: Any = [1.0, 4.0]
    first_parent: Any = list(parent)
    first_donor: Any = list(donor)
    second_parent: Any = list(parent)
    second_donor: Any = list(donor)
    tools.rng.seed(6)
    (first,) = tools.mut_iso_line(first_parent, first_donor, iso=0.0, sigma=0.0)
    tools.rng.seed(6)
    (second,) = tools.mut_iso_line(second_parent, second_donor, iso=0.0, sigma=0.0)
    assert first == second
