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
