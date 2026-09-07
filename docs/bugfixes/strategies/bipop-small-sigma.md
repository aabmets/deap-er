# BIPOP small-regime $\sigma$ ignored $\sigma_{\mathrm{large}}$

Hansen's BIPOP draws a small-regime step size
$\sigma = \sigma_0\cdot 10^{-2U[0,1]}$, so the sample lives in
$[0.01\,\sigma_0,\,\sigma_0]$. `sample_small_sigma` used a hardcoded
$2.0$ instead of `sigma_large`.

The default `sigma_large=2.0` hid the bug. A tighter box
(`sigma_large=0.25`) still drew $\sigma$ up to $2$.

**Fix.** Scale the draw by `sigma_large`.

**Validator.**
`tests/test_strategies/test_restart_ops.py::test_bipop_small_sigma_scales_with_sigma_large`
