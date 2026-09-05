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
from ._bm_landscape import bm_himmelblau, bm_schaffer, bm_schwefel
from ._bm_multimodal import bm_ackley, bm_bohachevsky, bm_griewank, bm_h1
from ._bm_rastrigin import bm_rastrigin, bm_rastrigin_scaled, bm_rastrigin_skewed, bm_shekel
from ._bm_unimodal import bm_cigar, bm_plane, bm_rand, bm_rosenbrock, bm_sphere

__all__ = [
    "bm_rand",
    "bm_plane",
    "bm_sphere",
    "bm_cigar",
    "bm_rosenbrock",
    "bm_h1",
    "bm_ackley",
    "bm_bohachevsky",
    "bm_griewank",
    "bm_schaffer",
    "bm_schwefel",
    "bm_himmelblau",
    "bm_rastrigin",
    "bm_rastrigin_scaled",
    "bm_rastrigin_skewed",
    "bm_shekel",
]
