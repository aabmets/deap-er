# Features 11-20

Items in this decade from the [roadmap overview](index.md).
Status and surfaces live on that table. Empty slots stay empty
until an item is numbered into this range.

---

## 11. SMS-EMOA

**What.** Environmental selection that discards the individual
with the smallest hypervolume contribution. Steady-state or
generational. `least_contrib` and `hypervolume` already delegate
to moocore.

**Today.** `sel_sms_emoa` performs SMS-EMOA environmental selection:
non-dominated sorting, then repeated removal of the least hypervolume
contributor on the critical front via ``least_contrib`` (moocore).
Use it generational on ``parents + offspring`` or steady-state on
``parents + [child]``. Optional ``ref_point`` follows the
``hypervolume`` / ``least_contrib`` minimization-space convention.

**Benefit.** The expensive part of SMS-EMOA is already done. The
algorithm is a loop over contributions. Completes the story that
the library computes hypervolume and should be able to *search*
with it, without a new dependency.

Related: [Operators](../../reference/operators.md).

---

## 12. MOEA/D and AGE-MOEA-II

**What.** Two many-objective selectors:

- **MOEA/D** — a weight-vector loop with Tchebycheff or PBI
  scalarization. NSGA-III already exposes
  `uniform_reference_points`; the missing piece is decomposition
  selection.
- **AGE-MOEA-II** — estimate front curvature (Newton–Raphson) and
  assign survival by geodesic distance. After MOEA/D: more code,
  more payoff when the front is not a simplex.

**Today.** `sel_moead` and `SelMOEADWithMemory` decompose with
Tchebycheff or PBI scalarization over `uniform_reference_points`
weight vectors. `sel_age_moea_2` and `SelAGE2WithMemory` estimate
front curvature with Newton–Raphson and break ties on the last
partial front with geodesic diversity.

**Benefit.** Many-objective users currently stop at NSGA-III.
MOEA/D is the standard decomposition method. AGE-MOEA-II is the
usual next step when NSGA-III’s simplex assumption fails on an
irregular front. Parity with the algorithms papers name, without
becoming a general MOO framework.

---

## 13. IPOP / BIPOP CMA restarts

**What.** A small algorithm next to `ea_generate_update` that
grows $\lambda$ and resets the CMA state on stagnation
(IPOP / BIPOP). It wraps the existing `Strategy` objects
(`Strategy`, `StrategyOnePlusLambda`, `StrategyMultiObjective`);
it does not add a new covariance-update variant.

**Today.** ``RestartStrategy`` wraps ``Strategy``, ``StrategySeparable``,
``StrategyOnePlusLambda``, and ``StrategyMultiObjective`` with IPOP
or BIPOP restart scheduling (BBOB-style stagnation, λ doubling, and
small-regime sampling). ``ea_generate_update_restarts`` runs until
the evaluation budget is spent. Box constraints from the inner
strategy are preserved. LM-CMA and learned step-size controllers
stay off the list. High-dimension Sep-CMA is [item 19](#19-sep-cma).

**Benefit.** This is how CMA is used on hard landscapes: enlarge
the population, reset the model, continue. ES users stop writing
the same wrapper. Secondary for prefix-tree GP; first-class for
continuous search.

Related: [Strategies](../../reference/strategies.md),
[Algorithms](../../reference/algorithms.md).

---

## 14. Linear-time duplicate count

**What.** Rewrite `duplicate_count` to $O(n)$ for hashable keys
(a `set`, or a sort when values are ordered but unhashable). The
function’s contract stays the same: how many individuals are
duplicates of an earlier one under an optional `key`.

**Today.** `duplicate_count` extracts keys once, then counts
$len(population) - distinct$ with a set when keys are hashable
($O(n)$), a sort when they are unhashable but mutually sortable
($O(n \log n)$), or list membership otherwise ($O(n^2)$). The
public contract is unchanged.

**Benefit.** Housekeeping that does not change the search language.
Callers who log uniqueness on a large population get a linear
scan instead of a quadratic one. No new operator.

Related: [Utilities](../../reference/utilities.md).

---

## 15. Heterogeneous crossover

**What.** `cx_heterogeneous` next to `mut_heterogeneous`: one
crossover callable per gene (or per slice) so a mixed encoding —
bit + int range + choice + boxed real — does not need a one-off
`mate()`. Each callable receives the pair of gene values and
returns the two replacements. The individual is modified in place.

**Today.** `cx_heterogeneous` dispatches one crossover per gene
(`callable(v1, v2) -> (v1', v2')`) or one existing `cx_*` per
slice. Both parents are modified in place. The
[mixed-encoding example](../../examples/genetic_algorithms/mixed_encoding.md)
registers uniform swap on the discrete prefix and
`cx_blend_bounded` on the boxed tail. `mut_heterogeneous` is
unchanged.

**Benefit.** Mixed genomes already have a mutator. Crossover is the
missing pair. Callers stop forking `mate()` every time the
encoding is heterogeneous.

**Scope.** A per-gene (or per-slice) dispatcher, not a catalog of
typed crossovers. Existing `cx_*` operators remain the callables
you pass in.

Related: [Operators](../../reference/operators.md).

---

## 16. Bounded Gaussian mutation

**What.** `mut_gaussian_bounded`: the same $N(\mu, \sigma)$ draw as
`mut_gaussian`, then a clamp (or a redraw) so each gene stays in
`[low, up]`. `low` / `up` may be scalars or per-gene sequences,
matching `mut_polynomial_bounded`.

**Today.** `mut_gaussian_bounded` applies the same $N(\mu, \sigma)$
add as `mut_gaussian`, then clamps each mutated gene into
`[low, up]`. Bounds may be scalars or per-gene sequences. An empty
interval (`up <= low`) is skipped. Unmutated genes are left as they
are. `mut_gaussian` is unchanged.

**Benefit.** The usual real-coded GA mutation when the search space
is a box and polynomial mutation is not wanted. Out-of-box genes
are the class of defect that made unbounded SBX write NaN or
complex values.

**Scope.** One operator. No new bound-repair framework.

Related: [Operators](../../reference/operators.md).

---

## 17. Differential evolution operators

**What.** The DE trial-vector recipe as toolbox functions, for
example `mut_de` / `cx_de` (or one `de_trial`). For a parent,
pick three others $a$, $b$, $c$, write

$$
y_i = a_i + F\,(b_i - c_i)
$$

on a binomial subset of genes (rate `CR`, at least one gene
forced), and let the caller keep $y$ when it is better. Register
them like any other variation operator.

**Today.** `mut_de` writes a DE/rand/1/bin trial in place:
$y_i = a_i + F\,(b_i - c_i)$ on a binomial subset of genes
(rate `cx_prob`, at least one gene forced). The caller supplies
donors $a$, $b$, $c$ and keeps $y$ when it is better. Optional
`low` / `up` clamp written genes. There is no `ea_de`. PSO stays
an example. The basic recipe is in
[the DE examples](../../examples/genetic_algorithms/diff_evo.md).

**Benefit.** DE is the algorithm that is still a recipe. Operators
close that without a new algorithm family.

**Scope.** Variation operators only. No first-class DE or PSO
algorithm, no adaptive $F$ / `CR` controller.

Related: [Operators](../../reference/operators.md).

---

## 18. Constraint-dominance selection

**What.** Feasibility-first comparison (Deb) for environmental
selection, usable from `sel_nsga_2` or as a thin wrapper. Three
rules: feasible beats infeasible; two feasibles use ordinary
Pareto / crowding on objectives; two infeasibles prefer the
smaller constraint violation. The caller supplies a feasibility
flag or a violation amount. Fitness values are not rewritten.

**Today.** `constraint_dominates` is Deb's constrained-domination
rule. `sel_nsga_2` accepts optional ``feasible=`` and
``violation=`` callables and ranks with that rule: feasible before
infeasible, ordinary Pareto / crowding among feasibles, smaller
violation among infeasibles. Omitted kwargs keep unconstrained
NSGA-II. Fitness values are not rewritten. `DeltaPenalty` and
`ClosestValidPenalty` remain the penalty path.

**Benefit.** Constrained NSGA-II users stop inventing a penalty
scale. Penalties remain the other path.

**Scope.** One comparison rule (and the selector that uses it).
Not a constraint framework, not a catalog of violation metrics.

Related: [Operators](../../reference/operators.md),
[Utilities](../../reference/utilities.md).

---

## 19. Sep-CMA

**What.** A separable CMA strategy next to `Strategy`: the
covariance $C$ stays diagonal — one variance per gene, no learned
correlations. Memory and update drop from $O(n^2)$ / $O(n^3)$ to
$O(n)$. Same `generate` / `update` surface, including the existing
`low` / `up` box. `RestartStrategy` can wrap it.

**Today.** `StrategySeparable` learns a length-$n$ diagonal $C$
(Ros and Hansen, 2008). Default `rank_one` / `rank_mu` are the
`Strategy` defaults scaled by $(n + 2) / 3$. `generate` / `update`,
`low` / `up`, and `RestartStrategy` match the other CMA strategies.
`Strategy` still owns the full matrix.

**Benefit.** High-dimension continuous search keeps a CMA-shaped
strategy when a full matrix no longer fits. Weaker when variables
interact; the right tool when they are roughly independent or $n$
is large.

**Scope.** Diagonal $C$ only. No LM-CMA, VkD-CMA, or learned
step-size controller until a user hits a documented wall after
this.

Related: [Strategies](../../reference/strategies.md).

---

## 20. CVT / unstructured MAP-Elites

**What.** A second quality-diversity archive next to `GridArchive`.
A **CVT** archive places $k$ centroids in descriptor space
(usually from a sample of possible behaviors) and assigns each
individual to the nearest centroid. An **unstructured** archive
skips a fixed tessellation and keeps elites by distance: add if
far enough from existing members, or replace a neighbor. Same
`add` / `random_elites` surface; `ea_map_elites` can take either
archive.

**Today.** `CvtArchive` assigns each descriptor to the nearest of
$k$ centroids (caller-supplied, or `cvt_centroids` / `from_samples`
via k-means). `UnstructuredArchive` adds a candidate when it is far
enough from every elite, otherwise replaces the nearest neighbor if
it is strictly fitter; optional `max_elites` caps growth.
`ea_map_elites` accepts either archive through the same
`add` / `random_elites` / `stats` surface as `GridArchive`. Fitness
stays on `ind.fitness`; the caller still supplies the descriptor.

**Benefit.** The usual next MAP-Elites data structure when
descriptors are irregular. Diversity stays in *behavior*, not
only in fitness. The caller still supplies the descriptor.

**Scope.** Archive data structures, not learned QD / meta-BBO
(see [Not planned](not_planned.md)). Fitness stays on
`ind.fitness`.

Related: [Records](../../reference/records.md),
[item 8](features_1_10.md#8-quality-diversity-archive).

---
