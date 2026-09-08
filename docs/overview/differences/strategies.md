# Evolution strategies

1. Every CMA strategy accepts `low` / `up` (scalar or length-`dim`)
   and a `bound_mode` of `"clip"` or `"resample"`, covering the
   [box-constrained CMA][deap-500] request. Bounds are applied to the
   sampled vector before `ind_init`. Resample rejects the whole
   vector; after `resample_limit` failed draws the offspring is
   clipped so `generate` always returns `lamb` individuals. Both
   modes are constraint-handling approximations: the CMA update then
   treats the repaired point as the sample.
2. MO-CMA's rank-one covariance update gates on $\lVert w \rVert$,
   not on `w.max()`. A negative evolution path no longer skips the
   update. When $w \approx 0$ the factors still scale by
   $\sqrt{\alpha}$.
3. The stall-case $\alpha$ on MO-CMA includes the $c_{\mathrm{cov}}$
   factor, so repeated stall generations do not inflate the
   covariance.
4. MO-CMA writes one step-size trial per parent when
   $\lambda \neq \mu$, so several children of the same parent do not
   stack $\sigma$ updates. `generate` samples every parent if any
   parent fitness is invalid; `update` ranks only valid fitnesses.
   When fewer than $\lambda$ parents remain, children are sampled
   from the available set instead of indexing past the last parent.
5. `RestartStrategy` wraps standard, separable, $(1+\lambda)$, or
   MO-CMA with IPOP or BIPOP restart scheduling. `ea_generate_update_restarts`
   runs the generate/update loop and logs restart regime, population
   size, and evaluation budget each generation.
6. `sample_centroid` replaces a non-finite box end with the
   unbounded default $[-5, 5]$, so a one-sided bound no longer
   writes `inf` or `NaN` as an IPOP/BIPOP restart mean.
7. Restart `TolFun` needs Hansen's $10 + 30n/\lambda$ history, then
   stops only if that window's best-of-generation range is below
   `tol_fun`. Two equal generation-bests no longer terminate at
   generation 2.
8. `StrategySeparable` is separable CMA (Ros and Hansen, 2008): $C$
   stays a length-$n$ diagonal, sampling and the update are $O(n)$,
   and default $c_1$ / $c_\mu$ are the full-matrix rates scaled by
   $(n + 2) / 3$. The `generate` / `update` surface, including
   `low` / `up`, matches `Strategy`. `RestartStrategy` can wrap it.
9. $(1+\lambda)$ CMA does not treat an unevaluated parent as worse
   than every offspring. After `reset_state` / an IPOP restart the
   first update adopts the best child without a fake
   $p_{\mathrm{succ}}=1$ step-size blow-up.
10. BIPOP small-regime $\sigma$ is $\sigma_{\mathrm{large}}\cdot
    10^{-2U[0,1]}$, not a hardcoded $2.0$. A custom first-run step
    size no longer launches small restarts in $[0.02, 2]$.
11. Default $\mu$ is at least $1$ when $\lambda\ge 1$.
    $\mu=\lfloor\lambda/2\rfloor$ is no longer $0$ at $\lambda=1$,
    so `Strategy` / `StrategySeparable` and a `RestartStrategy`
    leftover budget of one evaluation no longer raise
    `ZeroDivisionError` on an empty weight vector.
12. `Strategy.compute_params` rebuilds $C$, $B$, and $D$ only on
    first init or when `cm_init` is in `kwargs`. A later
    `compute_params(offsprings=…)` no longer wipes the learned
    covariance.
13. A leftover-budget CMA batch keeps
    $\mu=\min(\mu_{\mathrm{prev}},\lambda)$.
    `resize_offsprings` no longer resets survivors to
    $\lfloor\lambda/2\rfloor$, so a $\lambda=8$, $\mu=4$ run
    with 3 evaluations left does not finish at $\mu=1$.

[deap-500]: https://github.com/DEAP/deap/issues/500
