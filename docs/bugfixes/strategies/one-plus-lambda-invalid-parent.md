# $(1+\lambda)$ CMA treated an unevaluated parent as a failure

`StrategyOnePlusLambda.update` compared `parent.fitness` to every
offspring with tuple order. An invalid fitness is `()`, and
`() <= (value,)` is true, so every child counted as a success.

`RestartStrategy` / `reset_state` delete the new parent's fitness.
The first update after a restart therefore set $p_{\mathrm{succ}}=1$,
grew $\sigma$, and replaced the parent even when every offspring was
worse than the pre-restart parent would have been.

**Fix.** If the parent has no valid fitness, adopt the best evaluated
offspring and skip the Igel success-rate / $\sigma$ / $C$ step.

**Validator.**
`tests/test_strategies/test_cma_one_plus_lambda.py::test_update_invalid_parent_adopts_best_without_fake_success`
