# CMA $\lambda=1$ default $\mu=0$ divides by zero

Hansen's default $\mu=\lfloor\lambda/2\rfloor$ is $0$ when
$\lambda=1$. `apply_cma_hyperparams` then builds an empty weight
vector and computes $\mu_{\mathrm{eff}}=1/\sum w^2$, which is
`ZeroDivisionError`.

`Strategy(offsprings=1)` and `StrategySeparable(offsprings=1)`
crash in the constructor. `RestartStrategy` hits the same path
when the leftover evaluation budget is $1$: `resize_offsprings`
calls `compute_params(offsprings=1)` and the last `generate`
raises. Explicit `survivors=1` already works, so $\lambda=1$ is a
valid CMA degeneracy — only the default $\mu$ is wrong.

**Fix.** Default $\mu$ to at least $1$ when $\lambda\ge 1$.

**Validator.**
`tests/test_strategies/test_cma_standard.py::test_offsprings_one_defaults_to_one_survivor`
`tests/test_strategies/test_restart_edges.py::test_last_batch_of_one_completes_restart_budget`
