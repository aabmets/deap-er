# DEAP/deap issues ported into deap-er

Local notes for the [DEAP/deap](https://github.com/DEAP/deap) issues implemented in this repo on 2026-09-06. Source tracker: <https://github.com/DEAP/deap/issues>.

These are ports of the agreed **Should implement** and **Could implement** items. They are not upstream DEAP patches. Default Fitness comparison, `sel_nsga_2` crowding, and docs/README were left alone.

| DEAP | Title | Commit |
| ---: | --- | --- |
| [#655](https://github.com/DEAP/deap/issues/655) | `mutPolynomialBounded` returns NaN | [`b43d440eb5d77665f88bb448c8eeceb95846024a`](https://github.com/aabmets/deap-er/commit/b43d440eb5d77665f88bb448c8eeceb95846024a) |
| [#740](https://github.com/DEAP/deap/issues/740) | `cxSimulatedBinaryBounded` raises on complex `eta` math | [`b43d440eb5d77665f88bb448c8eeceb95846024a`](https://github.com/aabmets/deap-er/commit/b43d440eb5d77665f88bb448c8eeceb95846024a) |
| [#527](https://github.com/DEAP/deap/issues/527) | Bounded blend crossover | [`b43d440eb5d77665f88bb448c8eeceb95846024a`](https://github.com/aabmets/deap-er/commit/b43d440eb5d77665f88bb448c8eeceb95846024a) (operator), [`1db151fcdc408fae6f9bb1d86c17b1fa87d6dc25`](https://github.com/aabmets/deap-er/commit/1db151fcdc408fae6f9bb1d86c17b1fa87d6dc25) (export) |
| [#472](https://github.com/DEAP/deap/issues/472) | PMX only works for `{0..n-1}` | [`c25412af210f9d1074373a8b35a4b845c788fd9b`](https://github.com/aabmets/deap-er/commit/c25412af210f9d1074373a8b35a4b845c788fd9b) |
| [#755](https://github.com/DEAP/deap/issues/755) | Heterogeneous / per-gene mutation | [`1db151fcdc408fae6f9bb1d86c17b1fa87d6dc25`](https://github.com/aabmets/deap-er/commit/1db151fcdc408fae6f9bb1d86c17b1fa87d6dc25) |
| [#500](https://github.com/DEAP/deap/issues/500) | CMA lower/upper bounds | [`4465e9e89ebc0dc981a165eec26cd7dd406f3bc6`](https://github.com/aabmets/deap-er/commit/4465e9e89ebc0dc981a165eec26cd7dd406f3bc6) |
| [#383](https://github.com/DEAP/deap/issues/383) | Weighted GP primitive sampling | [`e4b9d05984595d4cf990233d866c92bd1d4619a3`](https://github.com/aabmets/deap-er/commit/e4b9d05984595d4cf990233d866c92bd1d4619a3) |
| [#644](https://github.com/DEAP/deap/issues/644) | 0-arity terminals format as `name` not `name()` | [`e4b9d05984595d4cf990233d866c92bd1d4619a3`](https://github.com/aabmets/deap-er/commit/e4b9d05984595d4cf990233d866c92bd1d4619a3) |
| [#24](https://github.com/DEAP/deap/issues/24) | Infix pretty-printer for GP trees | [`e4b9d05984595d4cf990233d866c92bd1d4619a3`](https://github.com/aabmets/deap-er/commit/e4b9d05984595d4cf990233d866c92bd1d4619a3) |
| [#321](https://github.com/DEAP/deap/issues/321) | Crowding distance on `wvalues` | [`b4abadb2b22a60e6bb93b0b9d079bdb510bd7365`](https://github.com/aabmets/deap-er/commit/b4abadb2b22a60e6bb93b0b9d079bdb510bd7365) |
| [#641](https://github.com/DEAP/deap/issues/641) / [#247](https://github.com/DEAP/deap/issues/247) | DCD tournament requires `k % 4 == 0` | [`b4abadb2b22a60e6bb93b0b9d079bdb510bd7365`](https://github.com/aabmets/deap-er/commit/b4abadb2b22a60e6bb93b0b9d079bdb510bd7365) |
| [#720](https://github.com/DEAP/deap/issues/720) | `MultiStatistics` chapters need different functions | [`3cb9555b62a78c18f69ad9eed290bbc90237bea5`](https://github.com/aabmets/deap-er/commit/3cb9555b62a78c18f69ad9eed290bbc90237bea5) |
| [#694](https://github.com/DEAP/deap/issues/694) | `Logbook.stream` should print a header when empty | [`3cb9555b62a78c18f69ad9eed290bbc90237bea5`](https://github.com/aabmets/deap-er/commit/3cb9555b62a78c18f69ad9eed290bbc90237bea5) |
| [#121](https://github.com/DEAP/deap/issues/121) | Serialize logbook to JSON | [`3cb9555b62a78c18f69ad9eed290bbc90237bea5`](https://github.com/aabmets/deap-er/commit/3cb9555b62a78c18f69ad9eed290bbc90237bea5) |
| [#350](https://github.com/DEAP/deap/issues/350) | Duplicate-count statistic | [`3cb9555b62a78c18f69ad9eed290bbc90237bea5`](https://github.com/aabmets/deap-er/commit/3cb9555b62a78c18f69ad9eed290bbc90237bea5) |
| [#426](https://github.com/DEAP/deap/issues/426) | Per-generation wall time | [`3cb9555b62a78c18f69ad9eed290bbc90237bea5`](https://github.com/aabmets/deap-er/commit/3cb9555b62a78c18f69ad9eed290bbc90237bea5) |
| [#750](https://github.com/DEAP/deap/issues/750) | Logger instead of `print` | [`3cb9555b62a78c18f69ad9eed290bbc90237bea5`](https://github.com/aabmets/deap-er/commit/3cb9555b62a78c18f69ad9eed290bbc90237bea5) |
| [#735](https://github.com/DEAP/deap/issues/735) | Pareto front every generation | [`3cb9555b62a78c18f69ad9eed290bbc90237bea5`](https://github.com/aabmets/deap-er/commit/3cb9555b62a78c18f69ad9eed290bbc90237bea5) |

Operator reproductions for #655, #740, #527, #472, and #755 live in `1db151f` with the public exports.

---

## #655 — `mutPolynomialBounded` returns NaN

**Issue.** [DEAP#655](https://github.com/DEAP/deap/issues/655). Polynomial mutation computed `delta` from a gene already outside `[low, up]`. The base of `** mut_pow` went negative, Python produced a complex or `nan`, and that value was written back onto the individual.

**Why it was an issue.** NSGA-II and other bounded real-coded GAs start or drift genes outside the documented box (tight per-gene bounds, earlier unbounded variation). A single out-of-box gene poisoned fitness and the rest of the run. `eta <= 0` also hits `1/(eta+1)`.

**What we did.** `mut_polynomial_bounded` now requires `eta > 0`, skips a gene when `xu <= xl`, and clamps the gene into `[xl, xu]` *before* any power. The Deb formula and the final clamp are unchanged, so in-box golden cases stay the same.

**Commit.** `b43d440eb5d77665f88bb448c8eeceb95846024a`

---

## #740 — `cxSimulatedBinaryBounded` raises on complex math

**Issue.** [DEAP#740](https://github.com/DEAP/deap/issues/740). Bounded SBX built `beta` from parent-to-bound gaps. Out-of-box parents (or a bad `eta`) made `beta` negative, `alpha` complex, and `<=` between `float` and `complex` raised `TypeError`.

**Why it was an issue.** Same class of bug as #655: the operator assumed every parent gene was already inside the box. Real populations are not. Empty intervals (`xu == xl`) also divide by zero.

**What we did.** Same `require_positive_eta` helper. Skip `xu <= xl`. Clamp both parents into `[xl, xu]` before the Deb powers; if they coincide after the clamp, leave the locus unchanged. In-box numeric results are unchanged. A skipped 50% draw can still leave an out-of-box gene (clamp is only on the power path).

**Commit.** `b43d440eb5d77665f88bb448c8eeceb95846024a`

---

## #527 — Bounded blend crossover

**Issue.** [DEAP#527](https://github.com/DEAP/deap/issues/527). `cxBlend` has no box. Design problems that already used `cxSimulatedBinaryBounded` wanted the same `low`/`up` on blend.

**Why it was an issue.** Blend draws past the parents by `alpha`. Without a clamp, children leave a constrained search space that the rest of the toolbox treats as a box.

**What we did.** Added `cx_blend_bounded(ind1, ind2, alpha, low, up)`: same gamma as `cx_blend`, then clamp each child gene. Empty intervals are skipped. Exported from `operators.py` / `tools`.

**Commits.** Operator: `b43d440eb5d77665f88bb448c8eeceb95846024a`. Public export and tests: `1db151fcdc408fae6f9bb1d86c17b1fa87d6dc25`.

---

## #472 — PMX only works for `{0..n-1}`

**Issue.** [DEAP#472](https://github.com/DEAP/deap/issues/472). `cxPartialyMatched` (and uniform PMX) built position maps as `p1[ind1[i]] = i`. That only works when alleles *are* the indices.

**Why it was an issue.** TSP-style and named permutations (`['a','b','c']`, city IDs) IndexError or silently write the wrong genes. `cx_ordered` had the same allele-as-index hole array (`holes1[ind2[i]]`). One root cause, three operators.

**What we did.** `_allele_maps` builds `dict` allele→index maps and raises `ValueError` on duplicates or mismatched gene sets. `match` uses those dicts. `cx_ordered` uses membership sets for the kept slice. `{0..n-1}` still works.

**Commit.** `c25412af210f9d1074373a8b35a4b845c788fd9b` (tests in `1db151fcdc408fae6f9bb1d86c17b1fa87d6dc25`)

---

## #755 — Heterogeneous / per-gene mutation

**Issue.** [DEAP#755](https://github.com/DEAP/deap/issues/755). Every stock mutator treats every gene the same (flip all bits, Gaussian all floats). Mixed individuals (bit + int range + choice) had no operator.

**Why it was an issue.** Mixed-representation GAs are common in design/config search. Users had to write a one-off mutator for every encoding.

**What we did.** `mut_heterogeneous(individual, mutators, mut_prob)`: `mutators[i](value) -> value`, one callable per gene. Raises if the lengths differ. Lives in its own module so `mut_various.py` stays under the file-length cap. Exported from `tools`.

**Commit.** `1db151fcdc408fae6f9bb1d86c17b1fa87d6dc25`

---

## #500 — CMA lower and upper bounds

**Issue.** [DEAP#500](https://github.com/DEAP/deap/issues/500). `Strategy.generate` sampled an unbounded Gaussian. The only documented workaround was a penalty on evaluate, or switching to the standalone `cma` package.

**Why it was an issue.** Most engineering CMA runs are box-constrained. Unbounded samples waste evaluations and confuse anyone coming from bounded SBX/poly.

**What we did.** `low` / `up` (scalar or length-`dim`) and `bound_mode` of `"clip"` (default) or `"resample"` on every CMA strategy. Applied to the sampled vector *before* `ind_init`. Resample rejects the **whole** vector (truncated MVN, not per-gene redraw). After `resample_limit` (default 100) failed draws, that offspring is clipped so `generate` always returns `lamb` individuals. `StrategyMultiObjective.generate` is wired separately because it does not call `sample_offspring`. A later `compute_params()` without bound kwargs keeps existing bounds. Docstrings state both modes are constraint-handling approximations: the CMA update then treats the repaired point as the sample.

**Commit.** `4465e9e89ebc0dc981a165eec26cd7dd406f3bc6`

---

## #383 — Weighted GP primitive sampling

**Issue.** [DEAP#383](https://github.com/DEAP/deap/issues/383). `generate` picked primitives uniformly. Users wanted `add_primitive(..., weight=4)` so some functions appear more often.

**Why it was an issue.** Uniform sampling over `{add, sin, tan, ...}` over-represents rare operators and under-represents the arithmetic core of symbolic regression.

**What we did.** `Primitive.weight` defaults to `1.0` so existing trees stay equal. `add_primitive(..., weight=1.0)` on typed and untyped sets; `weight <= 0` is rejected. `choose_weighted` uses `rng.choice` when all candidate weights are equal (same RNG stream as before). Only unequal weights walk a cumulative sum with `rng.random()`. Same helper in `mut_node_replacement` and `mut_insert`. Terminals stay uniform. No `rng.choices` (`rng.py` is at the 270-line hard cap).

**Commit.** `e4b9d05984595d4cf990233d866c92bd1d4619a3`

---

## #644 — Zero-arity callable terminals format as `name` not `name()`

**Issue.** [DEAP#644](https://github.com/DEAP/deap/issues/644). A 0-arity function registered as a terminal stringified as `no_input_func_1`. `compile` / `eval` then looked up the function object instead of calling it.

**Why it was an issue.** The Python compile path is `eval` of `str(tree)`. Without `()`, a 0-arity terminal is a name, not a call, so the compiled program is wrong or raises.

**What we did.** `Terminal.call_zero`. `add_terminal` sets it when the registered context value is a named callable. `format()` emits `name()`. ARG terminals, numbers, and `True`/`False` stay unwrapped. `from_string` still tokenizes on `()`; tokens remain `no_input_func_1`.

**Commit.** `e4b9d05984595d4cf990233d866c92bd1d4619a3`

---

## #24 — Infix pretty-printer for GP trees

**Issue.** [DEAP#24](https://github.com/DEAP/deap/issues/24). Trees only printed as prefix `add(x, y)`. Symbolic-regression users wanted `x + y`.

**Why it was an issue.** Prefix dumps are hard to read in logs and papers. This is display-only; it is not a second compiler.

**What we did.** New module `deap_er/private/programming/infix.py` with `tree_to_infix`. Maps `add`/`sub`/`mul`/`div`/`neg` and the `v*` numpy aliases to infix; unknown primitives stay `name(args)`. Exported from `gp.py`. Not added to `compilers.py` (already near the line cap).

**Commit.** `e4b9d05984595d4cf990233d866c92bd1d4619a3`

---

## #321 — Crowding distance on `wvalues`

**Issue.** [DEAP#321](https://github.com/DEAP/deap/issues/321). `assignCrowdingDistance` crowds on `fitness.values`. The request was to crowd on `wvalues` so scale/sign of weights affect spacing.

**Why it was an issue.** Deb’s NSGA-II uses raw objectives. Some users want weighted space when objective magnitudes differ. Changing the default would change every NSGA-II run.

**What we did.** `assign_crowding_dist(..., *, use_weights=False)`. Default stays `values`. `sel_nsga_2` is unchanged. Note: for nonzero per-objective scales the Deb formula is affine-invariant, so `use_weights=True` only changes distances when a weight is `0` (that objective is skipped) or the caller has customized `wvalues`.

**Commit.** `b4abadb2b22a60e6bb93b0b9d079bdb510bd7365`

---

## #641 / #247 — DCD tournament requires `k` divisible by 4

**Issue.** [DEAP#641](https://github.com/DEAP/deap/issues/641), [DEAP#247](https://github.com/DEAP/deap/issues/247). `selTournamentDCD` walks two shuffled copies in steps of 4. `k=1` or `k=9` raised `IndexError` in DEAP; deap-er already raised `ValueError` for `k % 4 != 0`.

**Why it was an issue.** NSGA-II examples use `k == len(pop)` and even sizes. Anyone who wanted `len(pop)-1` or an odd `k` could not use DCD selection.

**What we did.** Keep the original paired-shuffle path when `sel_count % 4 == 0` (same results as before). For other `k` in `1..len(individuals)`, run pairwise DCD tournaments on shuffled copies until `k` winners. `sel_count > len` is still `ValueError`. `sel_count <= 0` returns `[]`.

**Commit.** `b4abadb2b22a60e6bb93b0b9d079bdb510bd7365`

---

## #720 — `MultiStatistics` chapters need different functions

**Issue.** [DEAP#720](https://github.com/DEAP/deap/issues/720). `MultiStatistics.register` forwarded the same function to every chapter. Fitness min/max and a GP terminal-count dict cannot share `numpy.mean`.

**Why it was an issue.** The workaround was to register on each inner `Statistics` before wrapping, or to change every `ea_*` signature to take several stats objects.

**What we did.** `MultiStatistics.register(..., chapters=None)`. `None` still registers on every chapter. A `str` or iterable names a subset. `chapters` is keyword-only so it is not bound into the statistic function.

**Commit.** `3cb9555b62a78c18f69ad9eed290bbc90237bea5`

---

## #694 — `Logbook.stream` should print a header when empty

**Issue.** [DEAP#694](https://github.com/DEAP/deap/issues/694). An empty logbook’s `stream()` printed “empty” instead of the column banner. Generation 0 can take a long time; users wanted the table header immediately.

**Why it was an issue.** `build_header` also assumed there were data rows (`max` over an empty `str_matrix` crashed). After we printed a header on empty, a naive `start_index == 0` reprint doubled the banner. Setting `header_streamed` on *any* empty `stream()` then dropped the banner for a book that had no `header` set.

**What we did.** Empty logbook with a `header` prints that header. `header_streamed` is set only when a header was actually emitted (`self.header` or the book already has rows), so a bare `Logbook().stream()` still says empty and the first real `stream()` still prints columns. Header-only sizing uses column-name widths.

**Commit.** `3cb9555b62a78c18f69ad9eed290bbc90237bea5`

---

## #121 — Serialize logbook to JSON

**Issue.** [DEAP#121](https://github.com/DEAP/deap/issues/121). `json.dumps(logbook)` lost chapters and numpy scalars.

**Why it was an issue.** Downstream tools and notebooks want a portable table, not a dill checkpoint.

**What we did.** `Logbook.to_json` / `from_json` serialize entries, chapters, and `header`. Numpy scalars become `float`/`int` via `.item()`; leftover non-JSON values are stringified. Round-trip restores chapters.

**Commit.** `3cb9555b62a78c18f69ad9eed290bbc90237bea5`

---

## #350 — Duplicate-count statistic

**Issue.** [DEAP#350](https://github.com/DEAP/deap/issues/350). Users wanted the number of twin individuals per generation as a `Statistics` field.

**Why it was an issue.** Variety collapse is invisible if you only log mean fitness.

**What we did.** `duplicate_count(population, key=None)` in `metrics.py` (identity key by default): `len(population)` minus distinct keys. Users register it on `Statistics`. Already star-exported via `tools`.

**Commit.** `3cb9555b62a78c18f69ad9eed290bbc90237bea5`

---

## #426 — Per-generation wall time

**Issue.** [DEAP#426](https://github.com/DEAP/deap/issues/426). Papers want a time-per-generation curve. The algorithms did not record duration.

**Why it was an issue.** Users wrapped `ea_*` or timed the whole run and lost per-gen resolution.

**What we did.** Optional `log_time=False` on all four `ea_*` functions. `time.perf_counter()` around each generation, including gen 0 evaluation. Writes `duration` into the record. `"duration"` is added to the printed header **only** when `log_time` is True, so the default table stays `gen`, `nevals`.

**Commit.** `3cb9555b62a78c18f69ad9eed290bbc90237bea5`

---

## #750 — Logger instead of `print`

**Issue.** [DEAP#750](https://github.com/DEAP/deap/issues/750). `verbose` used `print`. Inside Docker or a service, stdout is often not the log sink.

**Why it was an issue.** `print` also interacts badly with a destructive `logbook.stream`: a second read is empty, so `print` plus `logger.info(logbook.stream)` would log nothing.

**What we did.** Optional `logger: logging.Logger | None = None` on the four `ea_*` functions. When `verbose`, call `stream()` **once**; `print` that string if `logger is None`, else `logger.info(...)`.

**Commit.** `3cb9555b62a78c18f69ad9eed290bbc90237bea5`

---

## #735 — Pareto front every generation

**Issue.** [DEAP#735](https://github.com/DEAP/deap/issues/735). `eaMuPlusLambda` plus a `ParetoFront` HoF only yields the *cumulative* front at the end. Users wanted the front of each generation.

**Why it was an issue.** A HoF/ParetoFront is cumulative. Deepcopying it does not give “the front of generation t”.

**What we did.** Optional `fronts: list | None = None` on all four `ea_*` functions. After the HoF update, if `fronts` is a list, append a **new** `ParetoFront` built from the current `population` (that generation’s survivors, not the HoF). Length is `gens+1` for algorithms with a gen-0 record, `gens` for `ea_generate_update`.

**Commit.** `3cb9555b62a78c18f69ad9eed290bbc90237bea5`
