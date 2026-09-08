# Push GP

Two-level program search: **tapes stay the laws**, Push evolves the
**loop around them**. Push never sees raw columns or `matrix[t]`.
It reads summaries and emits discrete actions that already exist
as toolbox calls. That is not a second public genome — prefix
trees plus the tape remain the program. A public Push / Cartesian
/ linear genome stays on [Not planned](not_planned.md).

This page is a backlog, not a schedule. Status matches the
library: shipped preconditions are **shipped**; the rest are
**planned**. Shipped items are not re-specified here; the item
column links the matching write-up on a [Features](index.md)
page. Items 30, 31, 37, and 38 are shipped on both pages.

| # | Item | Surface | Status |
|:--|:-----|:--------|:-------|
| P1 | [Batch tape evaluation](features_1_10.md#4-batch-tape-evaluation) (item 4) | `gp` | shipped |
| P2 | [Down-sampled and informed lexicase](features_1_10.md#5-down-sampled-and-informed-lexicase) (item 5) | `operators` | shipped |
| P3 | [Case-structured evaluation helper](features_1_10.md#6-case-structured-evaluation-helper) (item 6) | utilities | shipped |
| P4 | [Quality-diversity archive](features_1_10.md#8-quality-diversity-archive) (item 8) | `records` | shipped |
| P5 | [Growing primitive language](features_21_30.md#21-growing-primitive-language) (item 21) | `gp` | shipped |
| P6 | [Semantic search space](features_21_30.md#22-semantic-search-space) (item 22) | `gp`, `records` | shipped |
| P7 | [Co-evolving cases](features_21_30.md#23-co-evolving-cases) (item 23) | `operators`, `records` | shipped |
| P8 | [Memetic constants](features_21_30.md#24-memetic-constants) (item 24) | `gp`, `strategies` | shipped |
| P9 | [Streaming and island ecology](features_21_30.md#25-streaming-and-island-ecology) (item 25) | `algorithms` | shipped |
| P10 | [Program teams](features_21_30.md#26-program-teams) (item 26) | `operators` | shipped |
| P11 | [Policy observation schema](#p11-policy-observation-schema) | `records`, `utilities` | shipped |
| P12 | [Policy action applicator](#p12-policy-action-applicator) | `algorithms` | shipped |
| P13 | [Held-out policy fitness](#p13-held-out-policy-fitness) | `records`, `operators` | shipped |
| P14 | [Action guards and cooldowns](#p14-action-guards-and-cooldowns) | `operators` | shipped |
| P15 | [Evaluation budget and eval cache](features_31_40.md#37-evaluation-budget-and-eval-cache) (item 37) | `algorithms`, `utilities` | shipped |
| P16 | [Causal lookback and suffix rescore](features_21_30.md#30-causal-lookback-and-suffix-rescore) (item 30) | `gp` | shipped |
| P17 | [Affine scaling and Lamarckian writeback](features_31_40.md#31-affine-scaling-and-lamarckian-writeback) (item 31) | `gp`, `utilities` | shipped |
| P18 | [Parallel RNG streams](features_31_40.md#38-parallel-rng-streams) (item 38) | `rng` | shipped |
| P19 | [Push GP as the loop](#p19-push-gp-as-the-loop) | `gp` (private policy) | shipped |

P1–P10 are shipped on the main table and still sit on this
path: the two-level loop *uses* them. P11–P14 are shipped
firewall pieces. P15–P18 are shipped.
P19 is last on purpose.

!!! note
    A linear policy or a fixed decision list on the same
    observe / action interface is a valid test that the
    firewall works. If that cannot beat
    `next_lexicase_cases` plus periodic tune on held-out
    quality per eval, stacks will not save the path.

---

## P1. Batch tape evaluation

**What.** Population-wide tape scoring against one packed
`(rows, columns)` matrix. The laws Push must not reimplement.

**Today.** Shipped as
[item 4](features_1_10.md#4-batch-tape-evaluation).
`interpret_tapes` and `evaluate_batch` are the only row
kernels on this path.

**Role.** Push never interprets a column. It may *request* a
rescore; the tape does the work.

---

## P2. Down-sampled and informed lexicase

**What.** Case-subset lexicase (`cases=`, informed sampling,
matrix filter). The geometry the policy chooses among.

**Today.** Shipped as
[item 5](features_1_10.md#5-down-sampled-and-informed-lexicase).

**Role.** A policy action sets the next `cases=` list. It
does not replace the selector.

---

## P3. Case-structured evaluation helper

**What.** Series → per-case errors with a `valid=` mask.

**Today.** Shipped as
[item 6](features_1_10.md#6-case-structured-evaluation-helper).

**Role.** Observations and held-out scores are reductions of
`case_errors`, not raw rows.

---

## P4. Quality-diversity archive

**What.** Behavior-binned elites (`add`, `random_elites`,
coverage / `qd_score`).

**Today.** Shipped as
[item 8](features_1_10.md#8-quality-diversity-archive).
[Item 20](features_11_20.md#20-cvt-unstructured-map-elites)
is the same surface on CVT / unstructured archives.

**Role.** Coverage and `qd_score` are legal observation
fields. Push does not write descriptors.

---

## P5. Growing primitive language

**What.** `promote_subtree` lifts a typed chunk into the pset.

**Today.** Shipped as
[item 21](features_21_30.md#21-growing-primitive-language).
Do not auto-promote every generation.

**Role.** Promote is a policy *action*, rate-limited by
[P14](#p14-action-guards-and-cooldowns).

---

## P6. Semantic search space

**What.** Tape outputs → moments, solve bits, projections,
nearest neighbors.

**Today.** Shipped as
[item 22](features_21_30.md#22-semantic-search-space).

**Role.** Solve bits and distances are summaries Push may
read. The pack stays behind `valid=` / `trust_matrix=`.

---

## P7. Co-evolving cases

**What.** `CaseExam` / `CaseExamPool`, `score_case_exams`,
`next_lexicase_cases`, `guard_case_exams`.

**Today.** Shipped as
[item 23](features_21_30.md#23-co-evolving-cases).

**Role.** The hand-written loop Push is allowed to replace.
`held_out` must remain unmarked by policy actions
([P13](#p13-held-out-policy-fitness)).

---

## P8. Memetic constants

**What.** Short boxed CMA / sep-CMA on ephemeral leaves and
`Window` ints.

**Today.** Shipped as
[item 24](features_21_30.md#24-memetic-constants).

**Role.** Tune is a costly action.
[P15](#p15-evaluation-budget-and-eval-cache) charges it to
`n_evals`.
[P17](#p17-affine-scaling-and-lamarckian-writeback) makes the
polish worth firing.

---

## P9. Streaming and island ecology

**What.** `step_islands` and the full-matrix append-only
rescore recipe.

**Today.** Shipped as
[item 25](features_21_30.md#25-streaming-and-island-ecology).
Legal dirty suffixes are
[item 30](features_21_30.md#30-causal-lookback-and-suffix-rescore).

**Role.** Migrate / deme pressure and “rescore now” are
actions. Legal dirty suffixes are [P16](#p16-causal-lookback-and-suffix-rescore).

---

## P10. Program teams

**What.** Greedy coverage of cases solved at 0.

**Today.** Shipped as
[item 26](features_21_30.md#26-program-teams).
Team scoring stays on the caller.

**Role.** A policy may pick `sel_count` or feed `sel_team`
a `cases=` list. It is not a per-row router.

---

## P11. Policy observation schema

**What.** `policy_observe(...)` returns one fixed, typed
vector. Allowed fields are summaries only: case-solve bits,
unsolved count, train vs held-out score, archive coverage /
`qd_score`, `nevals` used, rows seen, promoted-library size,
fitness-invalid flag, last action rejected. No `Array`, no
column slice, no `matrix[t]`.

**Today.** Shipped as
[`PolicyObservation`](../differences/utilities.md#10-policy-observation-schema)
and :func:`~deap_er.tools.policy_observe`. Summary helpers
coerce outputs from :func:`~deap_er.tools.case_errors`,
:func:`~deap_er.tools.score_case_exams`,
:class:`~deap_er.records.ArchiveStats`, and promoted-library
counters. Raw NumPy packs are rejected at the observation
boundary.

**Benefit.** This *is* the firewall. Without it, Push grows
a load-column opcode and the causal story is gone.

**Scope.** A helper and a documented layout. Not a new
genome. Not a domain metric.

Related: [P3](#p3-case-structured-evaluation-helper),
[P4](#p4-quality-diversity-archive),
[P6](#p6-semantic-search-space),
[P7](#p7-co-evolving-cases).

---

## P12. Policy action applicator

**What.** `apply_policy_action(action, ...)` maps a discrete
action onto existing callables only:

- set next `cases=` / pick an exam (`next_lexicase_cases`)
- tune / skip (`tune_ephemerals`)
- promote / skip (`promote_subtree`)
- invalidate and rescore (`evaluate_invalid` / `interpret_tapes`)
- migrate / pick deme pressure (`step_islands`)

**Today.** Shipped as
[item 29](../differences/records.md).
`SUPPORTED_POLICY_ACTIONS` documents the token schema. Skip
tokens are intentional no-ops; unknown tokens and missing
required kwargs are rejected without raising.

**Benefit.** Push emits actions, not trees. The applicator
is the thin loop that must not become a second `ea_*`
framework.

**Scope.** Schema plus dispatch. Evaluation stays on the
caller. No `step_program_search` that owns fitness.

Related: [P2](#p2-down-sampled-and-informed-lexicase),
[P5](#p5-growing-primitive-language),
[P8](#p8-memetic-constants),
[P9](#p9-streaming-and-island-ecology).

---

## P13. Held-out policy fitness

**What.** Policy individuals are scored only on a
caller-marked `held_out` exam (and/or a later slice the
action schema cannot name). Train-exam quality is an
observation, not the policy objective. Log the
generalization gap as its own Logbook chapter.

**Today.** Shipped as
[item 31](features_21_30.md#31-held-out-policy-fitness) on the
features page. ``policy_held_out_fitness`` scores only the
caller-marked ``held_out`` exam; ``guard_policy_fitness_exam``
refuses train or freshly mutated exams as the objective.
``policy_exam_scores`` and ``policy_observe`` keep train quality
as an observation. ``record_policy_generalization_gap`` logs
train, held-out, and gap as a ``generalization_gap`` Logbook
chapter.

**Benefit.** Otherwise Push evolves “make the exam easy.”

**Scope.** A scoring convention and a chapter. Chronological
meaning stays on the caller. Not a metric catalog.

Related: [P7](#p7-co-evolving-cases),
[P11](#p11-policy-observation-schema).

---

## P14. Action guards and cooldowns

**What.** Hard caps next to `guard_case_exams`: max promotes
per generation, max inner `tune` generations, minimum exam
size, promote cooldown, reject any action that exceeds
remaining `n_evals`. A rejected action is an observation,
not a crash. A uniform random policy must run thousands of
generations without melting the compile cache.

**Today.** `PolicyActionGuard` and `guard_policy_action`
enforce the caps on `apply_policy_action`. Rejected actions
return `PolicyActionResult(rejected=True)` without raising.
Call `begin_generation` each outer generation so per-generation
promote limits reset. `estimate_policy_action_evals` supplies
conservative budget checks for tune, rescore, and island steps.

**Benefit.** If a random policy cannot survive, do not add
Exec stacks.

**Scope.** Guards on [P12](#p12-policy-action-applicator)
only. Not a new selector.

Related: [P5](#p5-growing-primitive-language),
[P7](#p7-co-evolving-cases),
[P15](features_31_40.md#37-evaluation-budget-and-eval-cache).

---

## P15. Evaluation budget and eval cache

**What.** `n_evals=` on the shared loop and an `EvalCache`
keyed by expression plus matrix identity. Every tune,
rescore, and promote-induced recompile spends that budget.
Policy fitness is held-out quality *per eval*.

**Today.** Shipped as
[item 37](features_31_40.md#37-evaluation-budget-and-eval-cache).
`n_evals=` stops `ea_simple`, `ea_mu_plus_lambda`,
`ea_mu_comma_lambda`, and `ea_map_elites` when the evaluation
count is spent; generations remain the default.
`ea_generate_update_restarts` already stops on evaluations.
`EvalCache` wraps `evaluate` / `evaluate_batch` by expression
text or a caller key plus matrix identity and row count.
`promote_subtree` and `tune_ephemerals` drop matching fitness
keys when they invalidate the compile cache.

**Role.** Without this, a policy that tunes every generation
wins by spending. Do not evolve Push until actions are
metered.

---

## P16. Causal lookback and suffix rescore

**What.** `tape_lookback` plus a suffix rescore that matches
the full-matrix oracle.

**Today.** Shipped as
[item 30](features_21_30.md#30-causal-lookback-and-suffix-rescore).
`tape_lookback` is the certificate; `suffix_rescore` writes
the dirty suffix onto a cached prefix and matches the
full-matrix oracle.

**Role.** A policy may say “new rows; rescore now.” The
lookback bound is what makes that action legal.

---

## P17. Affine scaling and Lamarckian writeback

**What.** Keijzer \(a + b\,f(x)\) before or as writeback
from `tune_ephemerals`.

**Today.** Shipped as
[item 31](features_31_40.md#31-affine-scaling-and-lamarckian-writeback).

**Role.** Makes tune actions polish a law instead of
fighting intercept and slope. Not a firewall piece; ship
before Push if `tune` is in the action set.

---

## P18. Parallel RNG streams

**What.** Independent, seedable worker streams that still
reproduce.

**Today.** Shipped as
[item 38](features_31_40.md#38-parallel-rng-streams).
`spawn_rng(seed, worker_id)` and `map_spawned` derive
independent worker streams. The process-wide generator stays
the default and stays checkpointable.

**Role.** Two populations (tapes and policies) under
`toolbox.map`. Skip only if both stay in-process.

---

## P19. Push GP as the loop

**What.** A **private** policy individual: a tiny Push
instruction set on `int` / `bool` / a short solve-bit
vector. `policy_observe` in, `apply_policy_action` out.
No column loads, no `Window` as a Push type, no public
`PushTree` on `gp` / `tools`. Tapes remain the only thing
`interpret_tapes` runs.

**Today.** Shipped as
[item 28](../differences/gp.md#28-private-push-gp-policy-loop).
`LinearPolicyProgram` and `PushPolicyProgram` live under
``deap_er.private.programming`` only — not on ``gp`` or ``tools``.
``step_policy_loop`` wires ``policy_observe`` in and
``apply_policy_action`` out. The in-tree Push interpreter reads
``int`` / ``bool`` / solve-bit summaries only; tapes remain the
only ``interpret_tapes`` target.

**Benefit.** Evolves the *operator of specialists* — which
exam, when to tune or promote, when to rescore, which
pressure — when a second prefix tree is too clumsy and a
hand-written `next_lexicase_cases` plus periodic tune is
too rigid.

**Scope.** Optional extra or in-tree interpreter, either
way behind the firewall. Not a second public genome. Not
per-row Push. Not `rolling_mean` as a Push instruction.
A linear policy on the same interface may ship first as
the acceptance test.

Related: [P11](#p11-policy-observation-schema),
[P12](#p12-policy-action-applicator),
[P13](#p13-held-out-policy-fitness),
[P14](#p14-action-guards-and-cooldowns),
[Not planned](not_planned.md).
