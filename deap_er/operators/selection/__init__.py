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
from .sel_helpers import assign_crowding_dist, uniform_reference_points
from .sel_lexicase import sel_epsilon_lexicase, sel_lexicase
from .sel_nsga_2 import sel_nsga_2
from .sel_nsga_3 import SelNSGA3WithMemory, sel_nsga_3
from .sel_spea_2 import sel_spea_2
from .sel_tournament import sel_double_tournament, sel_tournament, sel_tournament_dcd
from .sel_various import (
    sel_best,
    sel_random,
    sel_roulette,
    sel_stochastic_universal_sampling,
    sel_worst,
)

__all__ = [
    "SelNSGA3WithMemory",
    "assign_crowding_dist",
    "sel_best",
    "sel_double_tournament",
    "sel_epsilon_lexicase",
    "sel_lexicase",
    "sel_nsga_2",
    "sel_nsga_3",
    "sel_random",
    "sel_roulette",
    "sel_spea_2",
    "sel_stochastic_universal_sampling",
    "sel_tournament",
    "sel_tournament_dcd",
    "sel_worst",
    "uniform_reference_points",
]
