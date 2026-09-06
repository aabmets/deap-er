# Performance

deap-er is a rewrite, not a drop-in rename, but the same genomes and
GP expressions can be timed on both libraries. The chart below is
**relative speed**: DEAP/deap is the unit baseline (100%). A longer
bar means deap-er finished the same work in less wall time.

![Hot-path speed of aabmets/deap-er 3.0.0 relative to DEAP/deap 1.4.4](../images/hotpath-speedups.png)

Each bar is the mean of 50 timed runs after 2 warmups. Versions in
the title are the packages that produced the JSON, not whatever is
installed when the figure is redrawn.

| Case | Relative speed |
|:-----|---------------:|
| `nsga_convergence` n=40 | 45× |
| `compile_tree` 10 trees ×20 repeats | 17× |
| `sel_spea_2` n=160 k=80 | 5.2× |
| `sel_spea_2` n=80 k=40 | 4.6× |
| `clone_individual` n=60 | 3.5× |
| `sel_nsga_3` n=80 k=40 | 2.1× |
| `fitness.values` ×200 on n=80 | 1.6× |
| `fitness.dominates` pairwise n=80 | 1.6× |
| `compile_tree` 40 unique trees | 1.1× |
| `deepcopy` n=60 | 1.0× |
| `ea_simple` n=40 gens=8 | 0.84× |

Green is darker as the speedup grows past 100%. The gray bar is
below the DEAP baseline.

## What is faster

The largest gains sit on numeric or cached work that DEAP still does
in Python loops:

- **`nsga_convergence`** — pairwise distances go through SciPy
  `cdist` on fitness coordinates, not a Python double loop over
  genes.
- **`sel_spea_2` / `sel_nsga_3`** — dominance and niche assignment
  are vectorized. SPEA-II density rows stay the original
  upper-triangle layout so the RNG stream is unchanged.
- **`compile_tree` repeats** — the default `eval` backend is cached
  by expression text and context identity. Both libraries compile
  the same shared source strings. The first compile of 40 unique
  trees is slightly ahead of DEAP; repeating the same ten trees is
  about 17× faster.
- **`clone_individual`** — a shallow copy of a `list` / `array.array`
  individual plus a deepcopy of fitness. The default
  `Toolbox.clone` is still `deepcopy` (the near-parity bar). Register
  the fast clone when genes are a plain sequence and extra state is
  only fitness. See [Important differences](differences.md).
- **`fitness.values` reads** — the tuple is cached after the first
  assignment.
- **`fitness.dominates`** — one-, two-, and three-objective cases
  unpack `wvalues` and compare directly. Other arities use an
  indexed loop.

## What is slower

One case on this machine sits under 100%:

- **`ea_simple`** — about 0.84× DEAP (2.11 ms vs 1.76 ms on n=40,
  8 generations). The generational loop is almost entirely
  tournament selection. Each round calls `sel_random`, and each
  pick is a scalar `rng.choice`. Crossover and flip-bit mutation
  add more scalar `random` draws. DEAP uses CPython's `random`
  module for those. deap-er's process-wide RNG is a NumPy
  `Generator` facade (buffered uniforms and scalar integers) so
  one stream can be seeded, checkpointed, and matched by golden
  tests. A NumPy-backed draw is still more expensive than CPython
  `random`, so a tiny OneMax-style loop that spends its time in
  `choice` / `random` loses. The bench does not rewrite tournament
  to a batched sampler, because that would change the RNG stream.

That one case does not cancel the selection, clone, and compile
wins. A run that spends its time in SPEA-II, NSGA-III, or repeated
GP compile will see the chart's upper bars. A tiny OneMax loop that
is almost entirely tournament draws will look like `ea_simple`.

## How to reproduce

From the repo root, with the `dev` extra (DEAP and seaborn):

```text
uv run python tools/bench_hotpaths.py
uv run python tools/plot_hotpath_speedups.py -o docs/images/hotpath-speedups.png
```

The bench writes `reports/hotpath-bench.json` (gitignored). Both
libraries receive the same numeric genomes and the same GP
expression strings. First-time compile skips warmup and clears
deap-er's compile cache on every sample.
