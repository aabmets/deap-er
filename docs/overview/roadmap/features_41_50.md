# Features 41-50

Items in this decade from the [roadmap overview](index.md).
Status and surfaces live on that table. Empty slots stay empty
until an item is numbered into this range.

---

## 41. Case-structured generalization path

**What.** The default case-structured recipe the thin loops do
not wire: a caller-marked held-out exam, lexicase on the rest,
and a successive-halving schedule that spends cheap case
subsets first. Optional index helper for a last-fraction
holdout (`held_out_tail(n, fraction=)` or equivalent).
Chronological meaning, embargo, and expanding windows stay
on the caller.

**Today.** `CaseExam` / `CaseExamPool`, `next_lexicase_cases`,
informed down-sampling, `evaluate_columnar(..., reduce=False)`,
and `ea_policy` exist. Every example still builds the split
and the select callable by hand. There is no halving schedule
that charges partial exams to `n_evals`. Item 6 already
refused to own walk-forward; the
[index-only walk-forward builder](under_consideration.md#index-only-walk-forward-builder)
stays under consideration.

**Benefit.** One held-out slice plus lexicase on the train
cases is the usual difference between a curve-fit and a law
that still scores later. Halving stretches the same budget
across more individuals.

**Scope.** A documented path, a last-fraction holdout helper,
and a case-budget schedule. No timestamps. No domain loss.
Not [`step_program_search`](under_consideration.md#step_program_search).

Related: [item 5](features_1_10.md#5-down-sampled-and-informed-lexicase),
[item 6](features_1_10.md#6-case-structured-evaluation-helper),
[item 23](features_21_30.md#23-co-evolving-cases),
[item 28](features_21_30.md#28-dynamic-epsilon-and-downsample-schedule),
[item 33](features_31_40.md#33-evaluation-budget-and-eval-cache).

---

## 42. Memetic and affine leash

**What.** Defaults and a thin wrapper so `tune_ephemerals` and
affine scaling cannot spend the run. Darwinian affine (score
only) is the default path when writing case errors.
Lamarckian `write_affine_scale` stays opt-in. Tune uses a
small `n_gen`, charges the inner loop to remaining `n_evals`,
and is judged on a caller-marked held-out exam when one
exists.

**Today.** `MEMETIC_DEFAULT_N_GEN` and `MEMETIC_MAX_N_GEN`
document the recommended inner polish. `affine_case_errors`
fits Keijzer $a + b\,f(x)$ and writes Darwinian case errors
only. Lamarckian `write_affine_scale` stays opt-in.
`tune_ephemerals_budget` caps inner `n_gen` to remaining
`n_evals`, defaults to the small generation count, and judges
trials on a caller-marked held-out exam when one exists.
`cap_tune_n_gen` and `estimate_tune_ephemerals_evals` share
the same cost model as `PolicyActionGuard`. Raw
`tune_ephemerals` is unchanged. Policy caps stay on
`PolicyActionGuard`; `ea_simple` / `register_gp` callers use
the budget wrapper instead.

**Benefit.** Structure plus a short numeric polish is useful.
Unmetered tune and writeback on the train exam overfit
intercept, slope, and window lengths — the usual silent
failure next to a good tree shape.

**Scope.** Documented defaults, a Darwinian score helper, and
a budget-aware tune wrapper. Not a new CMA. Not a domain
metric. Policy caps stay on `PolicyActionGuard`.

Related: [item 24](features_21_30.md#24-memetic-constants),
[item 31](features_31_40.md#31-affine-scaling-and-lamarckian-writeback),
[item 33](features_31_40.md#33-evaluation-budget-and-eval-cache),
[item 41](#41-case-structured-generalization-path).

---

## 43. Team and archive recipe

**What.** A documented loop that keeps a *book* of programs,
not one elite: `semantic_project` → archive `add` →
`sel_team` on the case-solve matrix (or on occupied cells).
Optional thin helper that returns a team from archive elites
without rewriting member `fitness`.

**Today.** `semantic_project`, grid / CVT / unstructured
archives, and `sel_team` compose a documented caller loop in
the [columnar programs tutorial](../../tutorials/columnar_gp.md)
and the [team and archive example](../../examples/genetic_programming/team_archive.md).
`sel_team_archive` returns a team from occupied archive cells
without rewriting member `fitness`. Team scoring stays on the
caller. Not `step_program_search`.

**Benefit.** One elite on one window is how a train score dies
on the next slice. A team that covers different cases is what
lexicase and MAP-Elites were already pointing at.

**Scope.** A recipe and an optional helper on shipped
operators. Team scoring stays on the caller. Not a new
selector. Not [`step_program_search`](under_consideration.md#step_program_search)
— that wrapper also owns Slim, tune, and exams.

Related: [item 8](features_1_10.md#8-quality-diversity-archive),
[item 22](features_21_30.md#22-semantic-search-space),
[item 26](features_21_30.md#26-program-teams),
[item 29](features_21_30.md#29-novelty-selection-and-isoline),
[item 41](#41-case-structured-generalization-path).
