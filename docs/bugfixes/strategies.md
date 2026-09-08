# Strategies

Correctness fixes in CMA and multi-objective CMA.

---

## MO-CMA stall `alpha` dropped $c_{\mathrm{cov}}$

The stall branch wrote `1 - c_cov + cc * (2 - cc)` instead of
$(1 - c_{\mathrm{cov}}) + c_{\mathrm{cov}}\cdot cc\cdot(2-cc)$.
Default $n=5$ inflated the covariance ($\alpha \approx 1.425$ vs
published $\approx 0.967$).

**Fix.** Include the covariance learning rate in `alpha`.

**Validator.**
`tests/test_strategies/test_cma_multi_objective.py::test_stall_alpha_includes_covariance_learning_rate`

---

## Rank-one update gated on signed `w.max()`

The skip guard used the largest **signed** component of
$w = A^{-1}p_c$. An all-negative path (`‖w‖` large) skipped the
update, so $C' = \alpha C + \beta\,p_c p_c^\top$ was not
reflection-invariant.

**Fix.** Gate on magnitude (`numpy.max(numpy.abs(w)) > 1e-20`).

**Validator.**
`tests/test_strategies/test_cma_multi_objective.py::test_rank_one_update_runs_for_all_negative_path`

---

## Rank-one update dropped the $\alpha$ scale when $w \approx 0$

A numerically zero path skipped the whole body, including
$A \leftarrow \sqrt{\alpha}\,A$. Stall generations that should
contract $C$ left it unchanged.

**Fix.** When $w \approx 0$, still apply $A \leftarrow \sqrt{\alpha}\,A$
and $\mathrm{inv} \leftarrow \mathrm{inv}/\sqrt{\alpha}$.

**Validator.**
`tests/test_strategies/test_cma_multi_objective.py::test_rank_one_update_scales_when_path_is_zero`

---

## MO-CMA $\lambda \neq \mu$ updated $\sigma$ once per child

A parent with several offspring applied the Igel/Voss $p_{\mathrm{succ}}$
/ $\sigma$ update once per child instead of once per generation.

**Fix.** One trial per generation (`n_selected / n_from_parent`). The
one-child ($\lambda = \mu$) algebra is unchanged.

**Validator.**
`tests/test_strategies/test_cma_multi_objective.py::test_multi_child_parent_sigma_updates_once`

---

## `Strategy.compute_params` wiped the learned covariance

A later `compute_params(offsprings=…)` rebuilt $C$, $B$, and $D$ from
`cm_init`, discarding the learned covariance (and often $p_c$ / $p_\sigma$).

**Fix.** Rebuild $C$/$B$/$D$ only on first init or when `cm_init` is
in `kwargs`.

**Validator.**
`tests/test_strategies/test_cma_standard.py::test_compute_params_keeps_learned_c_unless_cm_init`

---

## One-sided box bounds sent restart centroids to inf/NaN

`sample_centroid` drew `rng.uniform(lo, hi)` after `_bound_arrays`
filled a missing end with $\pm\infty$. `uniform(0, \infty)` is
`inf`; `uniform(-\infty, 1)` is `NaN`. `RestartStrategy.restart()`
wrote that vector as the new mean.

**Fix.** Replace a non-finite end with the unbounded default
$[-5, 5]$, and expand a collapsed axis by that box width.

**Validator.**
`tests/test_strategies/test_cma_restart.py::test_restart_centroid_one_sided_bounds_stay_finite`

---

## MO-CMA `generate` assumed `len(parents) = λ = μ`

`update` keeps only candidates with valid fitness. When fewer than
`μ` are valid, the parent set shrinks. The next `generate` still
did `parents[i]` for `i in range(λ)` whenever `λ = μ`, and raised
`IndexError`.

**Fix.** Pair one child per parent only when enough parents exist.
Otherwise sample from the available parents (or return `[]` if
there are none).

**Validator.**
`tests/test_strategies/test_cma_multi_objective_bugfix.py::test_generate_after_partial_valid_when_lambda_equals_mu`

---

## Restart `TolFun` stopped after two equal generation-bests

`RunTracker._tol_fun_hit` compared only the first and last of a
2-generation span with a $10^{-12}$ relative tolerance. Two equal
generation-bests — common on a plateau or after box clipping —
terminated the run at generation 2 and forced an IPOP/BIPOP restart.

**Fix.** Require Hansen's $10 + 30n/\lambda$ history, then stop only
if that window's best-of-generation range is below `tol_fun`.

**Validator.**
`tests/test_strategies/test_cma_restart.py::test_run_tracker_equal_bests_do_not_stop_at_generation_two`

---

## $(1+\lambda)$ CMA treated an unevaluated parent as a failure

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

---

## BIPOP small-regime $\sigma$ ignored $\sigma_{\mathrm{large}}$

Hansen's BIPOP draws a small-regime step size
$\sigma = \sigma_0\cdot 10^{-2U[0,1]}$, so the sample lives in
$[0.01\,\sigma_0,\,\sigma_0]$. `sample_small_sigma` used a hardcoded
$2.0$ instead of `sigma_large`.

The default `sigma_large=2.0` hid the bug. A tighter box
(`sigma_large=0.25`) still drew $\sigma$ up to $2$.

**Fix.** Scale the draw by `sigma_large`.

**Validator.**
`tests/test_strategies/test_restart_ops.py::test_bipop_small_sigma_scales_with_sigma_large`

---

## CMA $\lambda=1$ default $\mu=0$ divides by zero

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
