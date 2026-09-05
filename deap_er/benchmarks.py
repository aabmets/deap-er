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
from .private.benchmarks.binary import (
    bm_chuang_f1,
    bm_chuang_f2,
    bm_chuang_f3,
    bm_royal_road_1,
    bm_royal_road_2,
)
from .private.benchmarks.bm_dtlz_1_4 import (
    bm_dtlz_1,
    bm_dtlz_2,
    bm_dtlz_3,
    bm_dtlz_4,
)
from .private.benchmarks.bm_dtlz_5_7 import bm_dtlz_5, bm_dtlz_6, bm_dtlz_7
from .private.benchmarks.bm_landscape import bm_himmelblau, bm_schaffer, bm_schwefel
from .private.benchmarks.bm_mo_classic import (
    bm_dent,
    bm_fonseca,
    bm_kursawe,
    bm_poloni,
    bm_schaffer_mo,
)
from .private.benchmarks.bm_multimodal import (
    bm_ackley,
    bm_bohachevsky,
    bm_griewank,
    bm_h1,
)
from .private.benchmarks.bm_rastrigin import (
    bm_rastrigin,
    bm_rastrigin_scaled,
    bm_rastrigin_skewed,
    bm_shekel,
)
from .private.benchmarks.bm_unimodal import (
    bm_cigar,
    bm_plane,
    bm_rand,
    bm_rosenbrock,
    bm_sphere,
)
from .private.benchmarks.bm_zdt import (
    bm_zdt_1,
    bm_zdt_2,
    bm_zdt_3,
    bm_zdt_4,
    bm_zdt_6,
)
from .private.benchmarks.moving_peaks import MovingPeaks, MPConfigs, MPFuncs
from .private.benchmarks.symb_regr import (
    bm_kotanchek,
    bm_rational_polynomial_1,
    bm_rational_polynomial_2,
    bm_ripple,
    bm_salustowicz_1d,
    bm_salustowicz_2d,
    bm_sin_cos,
    bm_unwrapped_ball,
)

__all__: list[str] = [
    "bm_royal_road_1",
    "bm_royal_road_2",
    "bm_chuang_f1",
    "bm_chuang_f2",
    "bm_chuang_f3",
    "bm_dtlz_1",
    "bm_dtlz_2",
    "bm_dtlz_3",
    "bm_dtlz_4",
    "bm_dtlz_5",
    "bm_dtlz_6",
    "bm_dtlz_7",
    "bm_schaffer",
    "bm_schwefel",
    "bm_himmelblau",
    "bm_kursawe",
    "bm_schaffer_mo",
    "bm_fonseca",
    "bm_poloni",
    "bm_dent",
    "bm_h1",
    "bm_ackley",
    "bm_bohachevsky",
    "bm_griewank",
    "bm_rastrigin",
    "bm_rastrigin_scaled",
    "bm_rastrigin_skewed",
    "bm_shekel",
    "bm_rand",
    "bm_plane",
    "bm_sphere",
    "bm_cigar",
    "bm_rosenbrock",
    "bm_zdt_1",
    "bm_zdt_2",
    "bm_zdt_3",
    "bm_zdt_4",
    "bm_zdt_6",
    "MovingPeaks",
    "MPConfigs",
    "MPFuncs",
    "bm_ripple",
    "bm_sin_cos",
    "bm_unwrapped_ball",
    "bm_kotanchek",
    "bm_salustowicz_1d",
    "bm_salustowicz_2d",
    "bm_rational_polynomial_1",
    "bm_rational_polynomial_2",
]
