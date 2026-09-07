# Algorithms

Correctness fixes in the algorithm drivers and variation operators.

---

## `ea_generate_update` never evaluates MO-CMA parents

`StrategyMultiObjective.generate` used to sample only from already-valid
parents. On generation 0 every parent is unevaluated, so the first
offspring batch was empty or stale and `update` never saw a scored
child.

**Fix.** `generate` skips `sort_non_dominated` and samples all parents
when any parent fitness is invalid. `update` ranks only
`ind.fitness.is_valid()` candidates, so the first generation promotes
evaluated offspring. The shared `ea_generate_update` driver is
unchanged.

**Validator.**
`tests/test_strategies/test_cma_multi_objective.py::test_invalid_parent_fitness_promotes_evaluated_offspring`

---

## `ea_generate_update` ignores `evaluate_batch`

`ea_simple`, `ea_mu_*`, and `ea_map_elites` score invalids through
`evaluate_invalid`, which calls `toolbox.evaluate_batch` when that
operator is registered. The generate-and-update drivers still went
through `toolbox.map(toolbox.evaluate, …)`, so a CMA run with a
vectorized or GPU batch evaluator never used it.

**Fix.** Both `ea_generate_update` and `ea_generate_update_restarts`
evaluate through `evaluate_invalid`. `nevals` is the number of
individuals that were actually scored.

**Validator.**
`tests/test_algorithms/test_ea_drivers.py::test_generate_update_uses_evaluate_batch_when_registered`

---

## `var_or` crashed when the parent pool had one individual

`rng.sample(population, 2)` needs two distinct parents. After
`ea_mu_comma_lambda` keeps a single survivor — a valid
$(1,\lambda)$ setting — the next generation raised
`ValueError` as soon as a crossover draw fired. The same crash
hits `ea_mu_plus_lambda` when $\mu = 1$ and `cx_prob > 0`.

**Fix.** When the pool has fewer than two individuals, clone the
only parent twice and mate those clones. Empty pools still fail
on `choice`.

**Validator.**
`tests/test_algorithms/test_variation.py::test_var_or_single_parent_with_crossover_does_not_crash`

---

## `ea_generate_update_restarts` discarded the last population on empty `generate`

`RestartStrategy.generate` returns `[]` when the eval budget is spent,
and a custom `generate` may use the same empty batch as a stop signal.
The driver assigned that empty list to `population` *before* breaking,
so a run that had already evaluated one or more batches returned `[]`
instead of the last scored generation. Hall of fame and the logbook
still had the work; the function result did not.

**Fix.** Bind the new batch only when it is non-empty. An empty
`generate` still stops the loop, but the returned population is the
last evaluated one (or `[]` if no generation ran).

**Validator.**
`tests/test_algorithms/test_ea_generate_update_restarts.py::test_empty_generate_keeps_last_evaluated_population`

---

## `ea_map_elites` crashed when compiling stats on an empty seed list

Generation 0 always compiled `stats` from `initial`. An empty seed is
a supported resume path: `_parent_pool` varies from a pre-filled
archive when `initial` is `[]`. The documented tutorial stats use
`max` / `numpy.max`, which raise `ValueError` on an empty reduction.
The same crash hit `record_generation` for an empty `ea_simple`
population.

**Fix.** Skip `stats.compile` when the individual list is empty.
Archive `coverage` / `num_elites` / `qd_score` still record. Later
generations with offspring still compile as before.

**Validator.**
`tests/test_algorithms/test_ea_map_elites.py::test_ea_map_elites_prefilled_archive_empty_initial_with_stats`
