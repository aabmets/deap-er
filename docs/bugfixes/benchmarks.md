# Benchmarks

Correctness fixes against published formulas and documented optima.

---

## Moving Peaks `pf1` peak shape and `ALT1` severity

Branke scenario 1 defines
$P = h / (1 + w \cdot \sum_i (x_i - p_i)^2)$. A first pass restored
Euclidean distance to match DEAP's wording; the published MPB
scenario 1 is the **squared** form, and that is what `MPFuncs.pf1`
uses now. The DEFAULT docstring matches that form.
`MPConfigs.ALT1["move_severity"]` is `1.5` (Branke scenario 2 / the
preset table), not `1.0`.

**Validators.**

- `tests/test_benchmarks/test_moving_peaks.py::test_pf1_squared_form_differs_from_euclidean`
- `tests/test_benchmarks/test_moving_peaks.py::test_pf1_uses_euclidean_distance_and_alt1_move_severity`

---

## DTLZ5 / DTLZ6 $f_1$ multiplies extra cosines

`_dtlz_helper_2` took $\prod \cos\theta$ over `individual[1:]`, so
distance variables that already feed $g$ were folded into $f_1$
again. $\sum f_i^2 = (1+g)^2$ broke whenever
`len(individual) > count`.

**Fix.** The $f_1$ product uses `individual[1:count-1]` only — the
same angular slice as $f_2 \ldots f_M$.

**Validator.**
`tests/test_benchmarks/test_multi_obj.py::test_dtlz5_f1_uses_only_the_angular_variables`

---

## Chuang F3 all-ones scored 31, not 40

The selector-1 branch kept `_inv_trap` on the rotated blocks, stopped
one block early (bit 38 unscored), and put the selector bit inside
the wrap trap. All-zeros scored 40; all-ones scored 31.

**Fix.** Selector-1 blocks use `_trap`. The 2-bit rotation covers
bits `2:38` in steps of 4 plus wrap `[38, 39, 0, 1]`. Bit 40 is
selector only. All-ones and all-zeros both score 40.

**Validator.**
`tests/test_benchmarks/test_binary.py::test_chuang_f3_optima_and_rotated_trap_blocks`

---

## Kotanchek used 3.2 instead of 1.2

The TEVC 2009 / GPTP target is
$1 / (1.2 + (x_2-2.5)^2)$. The implementation used `3.2`.

**Fix.** Both the function and the docstring table use `1.2`.

**Validator.**
`tests/test_benchmarks/test_symb_regr.py::test_kotanchek_uses_published_1_2_denominator`

---

## Royal Road R2 dropped the top-level schema

The hierarchy walk stopped before `n_order == len(individual)`, so
the full-length schema was never scored. Classic R2 (64 bits, order
8) could not reach 256.

**Fix.** Walk `n_order <= len(individual)` so the top schema is
included.

**Validator.**
`tests/test_benchmarks/test_binary.py::test_royal_road_2_classic_optimum_includes_top_schema`
