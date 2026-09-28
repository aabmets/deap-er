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
from .algorithms import *
from .algorithms import __all__ as _algorithms_all
from .benchmarks import *
from .benchmarks import __all__ as _benchmarks_all
from .operators import *
from .operators import __all__ as _operators_all
from .private.various_api import *
from .private.various_api import __all__ as _various_all
from .records import *
from .records import __all__ as _records_all
from .strategies import *
from .strategies import __all__ as _strategies_all

__all__ = list(
    dict.fromkeys(
        [
            *_algorithms_all,
            *_benchmarks_all,
            *_operators_all,
            *_records_all,
            *_strategies_all,
            *_various_all,
        ]
    )
)
