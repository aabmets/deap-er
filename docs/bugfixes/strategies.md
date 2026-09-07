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
