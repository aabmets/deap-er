# Constraints and utilities

1. `DeltaPenalty` and `ClosestValidPenalty` treat an `ndarray`
   `delta` or `distance` as a per-objective sequence. A 0-d array
   is broadcast; a 1-d vector is not passed to `itertools.repeat`.
2. `SortingNetwork.evaluate` copies each case, sorts the copy, and
   compares it to `sorted(original)`, so integer cases are not
   scored as bit-count patterns.
3. `case_errors` reduces aligned 1D `predicted` and `target` series
   into one mean-squared error per case segment. Accepts explicit
   half-open `(start, stop)` ranges or a boolean mask (one case per
   contiguous `True` run). Non-finite samples are skipped; optional
   `valid=` covers the `vwhere` warmup trap from columnar GP.
4. `duplicate_count` hashes a NumPy-array key by shape, dtype, and
   raw bytes. Equal ndarray individuals no longer raise `ValueError`
   from `sorted()` or list membership.
5. `nsga_convergence` and `nsga_diversity` read ``fitness.values``
   for reference individuals. A list genome is no longer treated as
   the objective vector when fitness is set.
6. `inv_gen_dist` uses the same fitness-first point extraction.
   Two fronts of individuals are no longer scored as gene lists.
7. `nsga_diversity` sorts the front by the first objective before
   Deb's $\Delta$ (a permutation of the same points no longer
   changes the value), and returns $1$ when the denominator is $0$
   (a single point, or several copies of one point with extremes
   at that point). It no longer raises `ZeroDivisionError`.
8. `SortingNetwork.draw` sizes the ASCII grid so empty and
   one-level networks no longer IndexError when writing wire
   labels or last-level spacers.
9. `spawn_rng(seed, worker_id)` and `map_spawned` give each
   mapped item an independent, seedable stream that does not
   collide with process-wide `tools.rng` and does not depend
   on pool scheduling ([user-provided streams][deap-75]). The
   parent generator stays checkpointable.
10. <a id="10-policy-observation-schema"></a>`PolicyObservation` and `policy_observe` define the fixed
   Push policy observation schema. Summary helpers coerce
   outputs from `case_errors`, `score_case_exams`,
   `ArchiveStats`, and promoted-library counters; raw NumPy
   packs and column slices are rejected at the boundary.
   `policy_exam_scores` splits train vs held-out exam
   difficulty without exposing `matrix[t]`. Not a genome and
   not a domain metric — the firewall before a private Push
   loop ([Push GP P11](../roadmap/push_gp.md#p11-policy-observation-schema)).
11. `sort_non_dominated` ranks only individuals with a finite,
    valid fitness. A mixed or all-invalid pool no longer raises
    `ValueError` or places unevaluated members on the first front.
    `sel_count <= 0` returns `[]`.
12. `nsga_diversity` returns $1$ for an empty front. `ordered[0]`
    no longer raises `IndexError`.
13. `nsga_convergence` and `inv_gen_dist` return $0$ when either
    point set is empty. `cdist` no longer raises `ValueError`
    on a 1-D empty array.
14. `affine_case_errors` fits Keijzer $a + b\,f(x)$ on the same
    `valid=` mask as `case_errors`, applies the scaled series, and
    returns per-case MSE without changing the tree. Darwinian
    scoring default; Lamarckian `write_affine_scale` stays opt-in.
15. `held_out_tail` and `train_head` return last-fraction holdout
    and train catalog indices. `case_generalization_pool` and
    `case_generalization_recipe` build a `CaseExamPool` with a
    caller-marked `held_out` exam. `make_lexicase_train_select`
    registers lexicase on train cases only. `case_halving_stages`,
    `evaluate_case_halving`, and `case_eval_charge` implement
    successive halving on train-catalog prefixes and charge partial
    exams in case-eval units for `n_evals=` budgeting. Chronological
    meaning stays on the caller. DEAP has no held-out lexicase recipe
    or case-budget halving schedule. See the
    [columnar GP tutorial](../../tutorials/columnar_gp.md).
16. `structural_meta_case_columns` and `structural_meta_case_weights`
    build cheap structural meta-cases (size, depth, unique opcodes,
    promote hits, non-finite fraction) to append to a packed case
    matrix for lexicase bloat and “always on” regularization without
    a second fitness weight. Lexicase selectors accept optional
    `fit_weights=` when the matrix is wider than ``fitness.values``.
    DEAP has no equivalent. See the
    [columnar GP tutorial](../../tutorials/columnar_gp.md).

[deap-75]: https://github.com/DEAP/deap/issues/75
