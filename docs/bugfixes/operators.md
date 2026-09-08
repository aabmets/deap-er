# Operators

Correctness fixes in crossover, mutation, selection, and migration.

---

## `mut_polynomial_bounded` wrote NaN or complex

Polynomial mutation computed $\delta$ from a gene already outside
`[low, up]`. The base of `** mut_pow` went negative, Python
produced a complex or `nan`, and that value was written back onto
the individual. `eta <= 0` also hits $1/(\eta+1)$.

**Fix.** Require `eta > 0`, skip a gene when `xu <= xl`, and clamp
the gene into `[xl, xu]` before any power. The Deb formula and the
final clamp are unchanged, so in-box cases stay the same.

**Validator.**
`tests/test_operators/test_mut_various.py::test_polynomial_bounded_out_of_box_stays_finite`

---

## Bounded SBX produced two lower children

`calc_c` always subtracted $\beta_q$, so $c_2$ used the upper-bound
$\beta_q$ with the lower-child formula. Every modified gene landed
below the parent midpoint.

**Fix.** $c_2$ is the plus child
$0.5\bigl((x_1+x_2)+\beta_q(x_2-x_1)\bigr)$ with $\beta_q$ from
$x_u-x_2$. $c_1$ still uses the minus side.

**Validator.**
`tests/test_operators/test_crossover.py::test_simulated_binary_bounded_second_child_uses_plus_side`

---

## `_slicer` aliased NumPy views

With `copy=False`, `temp = ind1[s]` is a live view. Assigning
`ind1[s] = ind2[s]` overwrote that view, so the second write put
parent 2's own values back. One-point and uniform on ndarray
individuals destroyed a parent instead of swapping.

**Fix.** Always materialize each slice (`copy()` / `list(...)`)
before assignment.

**Validator.**
`tests/test_operators/test_crossover.py::test_slicer_and_uniform_swap_numpy_without_aliasing`

---

## `cx_es_two_point_copy` did not copy the strategy

The genome swap forwarded `copy=True`. The strategy swap used the
default `copy=False`, so a NumPy $\sigma$ vector hit the same view
bug: parent 2's strategy was unchanged and parent 1's crossed
$\sigma$ values were lost.

**Fix.** Call `_slicer` on `strategy` with the same `copy` flag.

**Validator.**
`tests/test_operators/test_crossover.py::test_es_two_point_copy_swaps_numpy_strategy`

---

## `cx_messy_one_point` swapped one shared interval

Independent cuts `cxp1` and `cxp2` were passed to `_slicer` as
`(start, stop)`, so equal-length parents never changed length.

**Fix.** Swap tails independently:
`ind1[cxp1:], ind2[cxp2:] = list(ind2[cxp2:]), list(ind1[cxp1:])`.

**Validator.**
`tests/test_operators/test_crossover.py::test_messy_one_point_swaps_independent_tails`

---

## Length-1 shuffle / 1-point / 2-point raised `ValueError`

`randint(1, len-1)` is an empty range for a one-gene individual.
Fixed-length encodings of size 1 crashed instead of no-op.

**Fix.** `mut_shuffle_indexes`, `cx_one_point`, and `cx_two_point`
return the individual(s) unchanged when size is less than 2.

**Validator.**
`tests/test_operators/test_mut_various.py::test_mut_shuffle_indexes_length_one_is_noop`

---

## `sel_roulette` returned `[]` when all fitnesses were zero

`u = rng.random() * 0` and `0 > 0` never fired, so the mating pool
was empty. Negative fitness under-returned.

**Fix.** When the fitness sum is 0, sample uniformly `k` times.

**Validator.**
`tests/test_operators/test_selection.py::test_roulette_all_zero_fitness_returns_requested_count`

---

## Roulette and SUS sampled raw `values[0]`

Minimization (negative weight) preferred the **worst** individual.
Negative slices could also empty the wheel.

**Fix.** Build the wheel from `wvalues[0]` and shift it
non-negative. The `sum_fits == 0` path still falls back to uniform
choice.

**Validator.**
`tests/test_operators/test_sel_various.py::test_proportionate_selection_prefers_best_when_minimizing`

---

## `sel_tournament_dcd` ignored `sel_count`

The loop appended four winners per step of 4, so $k=1$ returned 4
and $k=9$ on a pool of 10 raised `IndexError`.

**Fix.** Require `sel_count % 4 == 0` for every $k$ and return
exactly $k$ individuals.

**Validator.**
`tests/test_operators/test_selection.py::test_tournament_dcd_returns_exact_count_and_rejects_non_multiple_of_four`

---

## SPEA-II density used a triangular, zero-padded row

The $k$-th neighbour distance included a padded self-distance of 0,
so isolated points were treated as infinitely dense.

**Fix.** Copy the full `sq_dist[i]` row and set the diagonal to
$+\infty$.

**Validator.**
`tests/test_operators/test_sel_spea_2.py::test_spea2_density_includes_isolated_point`

---

## `sel_spea_2(pop, k)` crashed on negative `k`

`len(chosen) > sel_count` is true for every first front when
`sel_count` is negative, so truncation ran `while size > sel_count`
past an empty archive and `del chosen[index]` raised `IndexError`.
`sel_nsga_2`, `sel_best`, and the other modern selectors already
return `[]` for `sel_count <= 0`.

**Fix.** Return `[]` when `sel_count <= 0` before ranking.

**Validator.**
`tests/test_operators/test_sel_spea_2.py::test_spea2_non_positive_count_returns_empty`

---

## `sel_spea_2([], k)` crashed

An empty pool entered `fill_from_density` and indexed an empty
fitness array. NSGA-II / NSGA-III already returned `[]`.

**Fix.** Return `[]` when the pool is empty.

**Validator.**
`tests/test_operators/test_sel_spea_2.py::test_spea2_empty_population_returns_empty`

---

## NSGA-III crashed when `sel_count` exceeded the pool

`select_from_niche` reduced an empty niche mask until it indexed
past the last front.

**Fix.** Stop when the last front is exhausted.
`sel_nsga_3` then returns the unique pool, matching `sel_nsga_2`.

**Validator.**
`tests/test_operators/test_sel_nsga_3.py::test_nsga3_oversize_sel_count_returns_unique_pool`

---

## NSGA-III intercepts mixed spaces; a constant objective yielded NaN

`find_intercepts` mixed translated and absolute coordinates. A
constant objective made `intercepts - best` zero and every niche
collapsed to NaN.

**Fix.** Return absolute intercepts on the success path
(`1/x + best`). A near-zero denominator in `associate_to_niche`
becomes `1.0`.

**Validator.**
`tests/test_operators/test_sel_nsga_3.py::test_nsga3_constant_objective_uses_absolute_intercepts`

---

## `mig_ring` used `list.index` (genotype equality)

Two equal genomes with different fitness evicted the wrong object
and duplicated the emigrant.

**Fix.** Find the vacancy with identity (`member is immigrant`).

**Validator.**
`tests/test_operators/test_migration.py::test_mig_ring_replaces_by_identity_not_genotype`

---

## `mig_ring(..., replacement=...)` aliased emigrants

The replacement individual was inserted by reference, so two demes
shared one object.

**Fix.** Clone emigrants before writing them into the destination
when `replacement` is set.

**Validator.**
`tests/test_operators/test_mig_ring.py::test_mig_ring_replacement_does_not_alias_across_demes`

---

## Identity replace crashed on duplicate emigrants

Looking up a vacancy after an earlier assignment had already
overwritten the object raised `ValueError`.

**Fix.** Resolve vacancy slots to indices first, then fill by those
indices.

**Validator.**
`tests/test_operators/test_mig_ring.py::test_mig_ring_sel_random_completes_with_duplicate_draws`

---

## PMX and ordered crossover indexed alleles as `{0..n-1}`

`cx_partially_matched`, uniform PMX, and `cx_ordered` built
position maps as `p1[ind1[i]] = i` (or a hole array keyed by
allele). That only works when alleles *are* the indices. Named
cities and other non-`{0..n-1}` encodings IndexError'd or
silently wrote the wrong genes.

**Fix.** `_allele_maps` builds `dict` allele→index maps and raises
`ValueError` on duplicates or mismatched gene sets. Ordered
crossover uses membership sets for the kept slice. `{0..n-1}`
still works.

**Validator.**
`tests/test_operators/test_cx_permutation.py::test_pmx_and_ordered_accept_letter_permutations`

---

## `cx_partially_matched` raised `ValueError` on empty permutations

`rng.randint(0, size - 1)` is an empty interval when both parents
have length 0. An empty permutation is a valid encoding — the
empty allele set — but PMX then asked the RNG for a second cut
on `[-1]`. `cx_ordered`, one-point, two-point, and shuffle already
no-op when size is less than 2. Uniform PMX already skipped the
loop.

**Fix.** After the shared-allele check, return the individuals
unchanged when `size < 2`.

**Validator.**
`tests/test_operators/test_cx_permutation.py::test_cx_partially_matched_empty_is_noop`

---

## `cx_ordered` raised `ValueError` on length-1 permutations

`rng.sample(range(size), 2)` needs two cut points. A one-gene
permutation (or two empty parents) is a valid encoding, but the
sample is larger than the population. One-point, two-point, and
shuffle already no-op when size is less than 2.

**Fix.** After the shared-allele check, return the individuals
unchanged when `size < 2`.

**Validator.**
`tests/test_operators/test_cx_permutation.py::test_cx_ordered_length_one_is_noop`

---

## NumPy integer bounds crashed `broadcast_param`

`isinstance(var, int | float)` rejects `numpy.int64`. Per-gene
operators then called `len()` on the scalar and raised `TypeError`.
`numpy.float64` already passed because it is a `float` subclass.
DEAP treats a non-sequence as a scalar, so `mutUniformInt(ind, 0,
np.int64(5), …)` works there.

**Fix.** After the Python `int | float` check, treat remaining
`numbers.Integral` values (NumPy integer scalars) as a broadcast
scalar. Sequences, including `ndarray`, still go through the length
check.

**Validator.**
`tests/test_operators/test_mut_various.py::test_mut_uniform_int_accepts_numpy_integer_bounds`

---

## `mig_ring` crashed when deme sizes or `mig_count` disagreed

A destination with more vacancies than the source had emigrants
indexed past the emigrant list (`IndexError`). `sel_random` with
`mig_count` larger than a deme then exhausted unused vacancy
indices (`StopIteration`). Unequal island sizes are a valid ring.

**Fix.** Stop claiming vacancies once every dest slot is taken.
Pair incoming emigrants with dest vacancies by `zip`, so extras
on either side are left in place. When `replacement` is omitted
and a home vacancy is not filled, clone that emigrant before
writing it into another island so two demes do not share one
object.

**Validator.**
`tests/test_operators/test_mig_ring.py::test_mig_ring_unequal_deme_sizes_completes`
and
`tests/test_operators/test_mig_ring.py::test_mig_ring_unequal_ring_does_not_alias_across_demes`

---

## NumPy `float32` bounds still crashed `broadcast_param`

The Integral patch left `numbers.Real` unimplemented.
`numpy.float32` is not a `float` subclass, so
`mut_gaussian_bounded(..., np.float32(0.0), np.float32(1.0), …)`,
`mut_polynomial_bounded`, and `cx_blend_bounded` still called
`len()` and raised `TypeError`. The inventory already claimed
Real scalars.

**Fix.** After `Integral`, treat remaining `numbers.Real` values
as a broadcast scalar (`float(var)`). 1-D bound sequences are
unchanged.

**Validator.**
`tests/test_operators/test_mut_various.py::test_bounded_operators_accept_numpy_float32_bounds`

---

## 0-d NumPy bounds crashed `broadcast_param`

NumPy scalars (`int64`, `float32`) already broadcast. A 0-d
`ndarray` such as `numpy.array(0.0)` is not `int`, `float`,
`Integral`, or `Real`, so `len()` raised `TypeError`. Callers that
wrap a scalar in `numpy.array` or `numpy.asarray` hit this path.

**Fix.** Treat a 0-d `ndarray` as a scalar (`var.item()`) and
broadcast it. 1-D bound sequences still go through the length
check.

**Validator.**
`tests/test_operators/test_bounds.py::test_bounded_operators_accept_numpy_0d_bounds`

---

## `sel_best` / `sel_worst` treated a negative `sel_count` as a slice

`sorted(...)[:sel_count]` uses Python's negative-index slice.
`sel_best(pop, -1)` returned every individual except the worst
instead of `[]`. `sel_random`, roulette, and NSGA-II already treat
`sel_count <= 0` as empty.

**Fix.** Return `[]` when `sel_count <= 0` before sorting.

**Validator.**
`tests/test_operators/test_sel_various.py::test_best_and_worst_non_positive_count_returns_empty`

---

## `mig_ring` aliased a duplicate emigrant in the destination

`sel_random` (and any selector that returns the same object twice)
wrote that object into two dest slots. Vacancy claiming already took
the next unused index, so both writes succeeded. Mutating one slot
then mutated the other. Source demes could alias the same way when
they received a duplicated emigrant.

**Fix.** Clone an emigrant when that object is already placed in the
destination this generation.

**Validator.**
`tests/test_operators/test_mig_ring.py::test_mig_ring_duplicate_emigrants_do_not_alias_in_dest`
