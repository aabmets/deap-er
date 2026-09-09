# Performance

The same genomes and GP expressions can be timed on both libraries.
The chart below is **mean wall time of one deap-er run** for each
component. Shared DEAP↔deap-er cases append
`(deap_er − deap) / deap × 100` in parentheses (negative means
deap-er was faster than DEAP). Unique deap-er features show the time
only. Figures and multipliers below come from
`reports/deaper_perf_bench.json` (deap-er 3.0.0 vs DEAP 1.4.4).

![Hot-path time of aabmets/deap-er versus DEAP/deap when shared](../images/deaper_perf_bench.png)

Bar color is the comparison class, not the time:

- **Green** — shared DEAP↔deap-er case where deap-er is faster.
  Darker green is a larger improvement versus DEAP, scaled to
  the biggest win on the chart.
- **Gray** — shared case where deap-er is slower than DEAP, or
  where percent change is missing.
- **Indigo** — deap-er-only feature (no DEAP counterpart).

Each bar is the mean of 50 timed runs after 2 warmups unless the JSON
`notes` field records a smaller repeat for a heavy CMA, MAP-Elites,
or Numba path. Versions in the title are the packages that produced
the JSON, not whatever is installed when the figure is redrawn.

Shared cases keep the same workloads as the previous DEAP comparison.

## What is faster than DEAP

The largest shared-case gains sit on numeric or cached work that DEAP
still does in Python loops. Relative speed is `deap_ms / deap_er_ms`
from the report:

- **`sel_lexicase`** (~68×) — case filtering runs on a packed
  `(n_individuals, n_cases)` matrix with NumPy boolean masks
  instead of per-case Python list comprehensions. Optional
  ``matrix=`` lets informed down-sampling and lexicase share one
  array per generation.
- **`nsga_convergence`** (~45×) — pairwise distances go through SciPy
  `cdist` on fitness coordinates, not a Python double loop over
  genes.
- **`sel_nsga_2` / `sel_spea_2` / `sel_nsga_3`** (~14× / ~5× / ~1.9×)
  — dominance and niche assignment are vectorized. NSGA-II ranks
  with `moocore.pareto_rank`. SPEA-II density rows stay the original
  upper-triangle layout so the RNG stream is unchanged. The larger
  SPEA-II case (n=160) is ~4.9×; n=80 is ~4.4×.
- **`compile_tree` repeats** (~13×) — the default `eval` backend is
  cached by expression text and context identity with LRU eviction
  at 1024 entries. Both libraries compile the same shared source
  strings. The first compile of 40 unique trees is slightly ahead of
  DEAP (~1.06×); repeating the same ten trees is about 13× faster.
- **`clone_individual`** (~3.6×) — a shallow copy of a `list` /
  `array.array` individual plus a deepcopy of fitness. The default
  `Toolbox.clone` is still `deepcopy` (~1.2× on list individuals).
  Register the fast clone when genes are a plain sequence and extra
  state is only fitness. See [Differences with DEAP](differences/gp.md).
- **`sel_tournament`** (~1.8×) — all contestant indices come from one
  `rng.integers(..., size=rounds * contestants)` draw. Winners are
  compared in Python. That stream differs from scalar `choice`.
- **`fitness.values` reads** (~1.6×) — the tuple is cached after the
  first assignment.
- **`fitness.dominates`** (~1.2×) — one-, two-, and three-objective
  cases unpack `wvalues` and compare directly. Other arities use an
  indexed loop.

## What is slower than DEAP

Two shared cases on this machine sit under the DEAP baseline:

- **`ea_simple`** — about 0.87× DEAP (2.13 ms vs 1.84 ms on n=40,
  8 generations). Isolated `sel_tournament` is ahead of DEAP.
  Flip-bit mutation now drains leftover uniforms with
  `rng.take_floats` (same stream as `random()`). Crossover still
  draws one scalar `rng.random` per mate-or-skip decision. DEAP
  uses CPython's `random` module for those. deap-er's process-wide
  RNG is a NumPy `Generator` facade (buffered uniforms and leftover
  integers) so one stream can be seeded, checkpointed, and matched
  by golden tests. A NumPy-backed uniform is still more expensive
  than CPython `random`, so a tiny OneMax-style loop that mixes
  those draws with variation loses.
- **`ParetoFront.update`** — about 0.97× DEAP (0.225 ms vs 0.218 ms
  on n=80). Near parity; the archive walk still uses the same
  `dominates` compare against a growing archive.

Those cases do not cancel the lexicase, selection, clone, and compile
wins. A run that spends its time in lexicase, SPEA-II, NSGA-II/III,
or repeated GP compile will see those shared bars improve. A tiny
OneMax loop that is almost entirely uniform draws in variation will
look like `ea_simple`.

## Unique features

The same bench also times deap-er-only capabilities from the
[differences inventory](differences/index.md) (features, not bugs): boxed
operators and CMA, SMS-EMOA / MOEA/D / AGE-MOEA-II, MAP-Elites
archives, island stepping, columnar GP tapes, SlimGP, and related
helpers. Pure docs or API cosmetics (`tree_to_infix`, `call_zero`,
empty Logbook header, logbook JSON, `ea_* logger`) are skipped.

## How to reproduce

From the repo root, with the `dev` extra (DEAP and seaborn):

```text
uv run python tools/perf_bench
uv run python tools/perf_bench -d reports
```

Both `deaper_perf_bench.json` and `deaper_perf_bench.png` land in the
same directory (`reports/` by default) and overwrite files already
there. `tools/bench_hotpaths.py` is a thin shim for the same entry.
The chart is written by importing `write_chart` directly (no
subprocess). To redraw the figure from an existing JSON, call
``plot_main`` from ``tools/perf_bench`` with the same ``-d``. Copy
`reports/deaper_perf_bench.png` to `docs/images/deaper_perf_bench.png`
to refresh the page figure.

Both libraries receive the same numeric genomes and the same GP
expression strings on shared cases. First-time compile skips warmup
and clears deap-er's compile cache on every sample.
