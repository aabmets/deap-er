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
from deap_er import tools


def _sphere(individual):
    return (sum(x * x for x in individual),)


def test_translate_rotate_and_scale_transform_the_individual():
    evaluate: Any = tools.Translation([1.0, 2.0])(_sphere)
    assert evaluate([1.0, 2.0]) == (0.0,)
    evaluate.translate([0.0, 0.0])
    assert evaluate([1.0, 2.0]) == (5.0,)

    rotate: Any = tools.Rotation(numpy.eye(2))(_sphere)
    assert rotate([1.0, 0.0]) == (1.0,)
    rotate.rotate(numpy.array([[0.0, 1.0], [1.0, 0.0]]))
    assert rotate([1.0, 0.0]) == (1.0,)

    scale: Any = tools.Scaling([2.0, 4.0])(_sphere)
    assert scale([2.0, 4.0]) == (2.0,)
    scale.scale([1.0, 1.0])
    assert scale([1.0, 2.0]) == (5.0,)


def test_noise_adds_per_objective_or_repeated_noise():
    def evaluate(_individual):
        return (1.0, 2.0)

    noisy: Any = tools.Noise([lambda: 0.5, None])(evaluate)
    assert noisy([0]) == (1.5, 2.0)
    noisy.add_noise(lambda: 1.0)
    assert noisy([0]) == (2.0, 3.0)

    scalar: Any = tools.Noise(lambda: 0.25)(lambda _ind: 4.0)
    assert scalar([0]) == (4.25,)


def test_bin2float_decodes_bit_blocks():
    def evaluate(individual):
        return tuple(individual)

    decoded: Any = tools.bin2float(0.0, 1.0, 2)(evaluate)
    # 11 -> 1.0, 00 -> 0.0
    assert decoded([1, 1, 0, 0]) == (1.0, 0.0)


def test_bin2float_decodes_boolean_bits_like_integers():
    def evaluate(individual):
        return tuple(individual)

    decoded: Any = tools.bin2float(0.0, 1.0, 2)(evaluate)
    assert decoded([True, True, False, False]) == (1.0, 0.0)
