# Utilities

Correctness fixes in metrics, constraint penalties, decoding, and
sorting networks.

---

## `nsga_diversity` depended on list order

Deb's $\Delta$ is defined on a front sorted along one objective.
The function used adjacent list entries as-is, so a permutation of
the same points changed $\Delta$ (perfectly uniform $0$ vs $0.5$).

**Fix.** Sort a copy of the front by the first objective before
computing $d_f$, $d_l$, and $d_t$.

**Validator.**
`tests/test_utilities/test_metrics.py::test_nsga_diversity_is_invariant_to_front_order`

---

## Single-point front returned $d_f + d_l$ instead of $\Delta = 1$

With $N=1$ both consecutive-distance terms vanish and Deb's formula
is $(d_f+d_l)/(d_f+d_l) = 1$. The code returned the unnormalized
numerator (for example $\sqrt{2}$).

**Fix.** Return `1.0` when `len(population) == 1`.

**Validator.**
`tests/test_utilities/test_metrics.py::test_nsga_diversity_for_a_single_point_is_one`

---

## `DeltaPenalty` treated an `ndarray` as a scalar

`numpy.ndarray` is not a `collections.abc.Sequence`, so a
per-objective vector went through `itertools.repeat`. The penalty
became a tuple of arrays and `Fitness.values` raised `TypeError`.

**Fix.** Treat an `ndarray` `delta` or `distance` as a per-objective
sequence (`numpy.asarray` + iterate).

**Validator.**
`tests/test_utilities/test_constraints.py::test_delta_penalty_accepts_ndarray_delta_and_distance`

---

## `ClosestValidPenalty` had the same `repeat()` wrap

Its distance check used the identical `isinstance(..., Sequence)`
test.

**Fix.** Same `ndarray` handling as `DeltaPenalty`.

**Validator.**
`tests/test_various/test_constraints.py::test_closest_valid_penalty_accepts_ndarray_distance`

---

## `bin2float` decoded `True`/`False` as `"True"`/`"False"`

`int("".join(map(str, values)), 2)` raised
`ValueError: invalid literal ... 'TrueTrue'`. `mut_flip_bit`
preserves `bool`, so a boolean individual crashed on evaluate.

**Fix.** Coerce each bit with `int(bool(v))` before joining.

**Validator.**
`tests/test_utilities/test_bm_decors.py::test_bin2float_decodes_boolean_bits_like_integers`

---

## `SortingNetwork.evaluate` was binary-only

Integer cases were scored against a bitstring lookup
(`ordered[sum(seq)]`) or crashed. Provided cases must be compared
to `sorted(original)`.

**Fix.** Copy each case, sort the copy, and compare it to
`sorted(original)`. Default bitstring `evaluate()` still reports 0
on a correct network.

**Validator.**
`tests/test_various/test_sorting_network.py::test_sorting_network_evaluate_provided_cases_against_sorted_original`

---

## `duplicate_count` crashed on NumPy individuals

NumPy arrays are unhashable, so the set scan raised `TypeError` as
intended. The sort fallback then called `sorted(keys)`, and ndarray
comparison raises `ValueError` (ambiguous truth value) instead of
`TypeError`. The list-membership fallback would hit the same error
on `value not in unique`.

**Fix.** Hash an ndarray key by ``(shape, dtype, tobytes())`` so the
set scan works. Two equal genomes count as one distinct key.

**Validator.**
`tests/test_various/test_metrics.py::test_duplicate_count_ndarray_individuals_count_twins`
