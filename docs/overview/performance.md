# Performance

deap-er is a rewrite, not a drop-in rename, but the same genomes can
be timed on both libraries. The chart below is **relative speed**:
DEAP/deap is the unit baseline (100%). A longer bar means deap-er
finished the same work in less wall time.

![Hot-path speed of aabmets/deap-er 3.0.0 relative to DEAP/deap 1.4.4](../images/hotpath-speedups.png)

Each bar is the mean of 30 timed runs after 2 warmups. Versions in
the title are the packages that produced the JSON, not whatever is
installed when the figure is redrawn.

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
  by expression text and context identity. The first compile of
  unique trees is about even with DEAP; repeating the same ten trees
  is several times faster.
- **`clone_individual`** — a shallow copy of a `list` / `array.array`
  individual plus a deepcopy of fitness. The default
  `Toolbox.clone` is still `deepcopy` (the near-parity bar). Register
  the fast clone when genes are a plain sequence and extra state is
  only fitness. See [Important differences](differences.md).
- **`fitness.values` reads** — the tuple is cached after the first
  assignment.

Green is darker as the speedup grows past 100%. Gray bars are below
the DEAP baseline.

## What is slower

Two cases on this machine sit under 100%:

- **`fitness.dominates`** — the comparison is now a single zip of
  `wvalues`, but DEAP still wins the microbenchmark.
- **`ea_simple`** — the generational loop is dominated by tournament
  selection. deap-er's process-wide RNG is a NumPy facade; CPython
  `random` is cheaper per `choice` / `randint`. The stream is kept
  for golden tests, so the bench does not rewrite tournament to a
  faster sampler.

Those two do not cancel the selection and metric wins. A run that
spends its time in SPEA-II, NSGA-III, or repeated GP compile will
see the chart's upper bars. A tiny OneMax loop that is almost
entirely tournament draws will look more like `ea_simple`.

## How to reproduce

From the repo root, with the `dev` extra (DEAP and seaborn):

```text
uv run python tools/bench_hotpaths.py
uv run python tools/plot_hotpath_speedups.py -o docs/images/hotpath-speedups.png
```

The bench writes `reports/hotpath-bench.json` (gitignored). Both
libraries receive the same numeric genomes. First-time compile skips
warmup and clears deap-er's compile cache on every sample.
