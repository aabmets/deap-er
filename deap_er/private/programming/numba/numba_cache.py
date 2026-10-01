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
import os
import sys
from pathlib import Path

__all__: list[str] = ["ensure_numba_cache_dir"]


def ensure_numba_cache_dir() -> None:
    """Point Numba at a writable on-disk cache when none is configured.

    The default location is an absolute path under the user's cache
    directory so spawned workers with a different working directory
    still share the same on-disk JIT cache. It also applies when numba
    was imported before this call, and a ``NUMBA_CACHE_DIR`` set after
    numba was imported is picked up too.
    """
    configured = os.environ.get("NUMBA_CACHE_DIR")
    if configured:
        config = sys.modules.get("numba.core.config")
        if config is not None and getattr(config, "CACHE_DIR", configured) != configured:
            # Numba read the variable at import, before the host set it.
            config.reload_config()
        return
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        cache = Path(xdg_cache) / "deap-er" / "numba"
    else:
        cache = Path.home() / ".cache" / "deap-er" / "numba"
    cache = cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache)
    config = sys.modules.get("numba.core.config")
    if config is not None:
        # Numba reads the variable at import. A consumer kernel module
        # usually imports numba first, so refresh its config here.
        config.reload_config()
