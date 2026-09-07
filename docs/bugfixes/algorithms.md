# Algorithms

Correctness fix in the generate-and-update driver.

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
