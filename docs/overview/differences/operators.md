# Operators and selection

These items were open on the DEAP tracker, or were defects inherited
from the original sources.

1. [`mut_polynomial_bounded`][deap-655] clamps each gene into
   `[low, up]` before the Deb powers and requires `eta > 0`, so an
   out-of-box gene no longer writes NaN or a complex value.
2. [`cx_simulated_binary_bounded`][deap-740] uses the same clamp and
   `eta` guard. Out-of-box parents no longer raise `TypeError` on
   complex arithmetic. The second child uses the opposite sign of
   $\beta_q$, matching Deb / DEAP (`c2` is the upper child, not a
   second lower child).
3. [`cx_blend_bounded`][deap-527] is the boxed form of blend
   crossover: the same $\gamma$ draw as `cx_blend`, then a clamp on
   each child gene.
4. [`cx_partially_matched`][deap-472], uniform PMX, and
   `cx_ordered` map alleles by value. Permutations of named cities or
   other non-`{0..n-1}` encodings no longer IndexError or silently
   write the wrong genes.
5. [`mut_heterogeneous`][deap-755] applies one mutator per gene, so a
   mixed encoding (bit + int range + choice) does not need a
   one-off mutator. `cx_heterogeneous` is the matching mate
   dispatcher: one callable per gene, or one existing `cx_*` per
   slice.
6. [`assign_crowding_dist`][deap-321] can crowd on `wvalues` via
   `use_weights=True`. The default, and `sel_nsga_2`, still use raw
   `values`.
7. [`sel_tournament_dcd`][deap-641] accepts any
   `1 ≤ k ≤ len(individuals)`. When `k` is a multiple of 4 the
   original paired-shuffle path is used; other counts run pairwise
   contests until `k` winners are collected. `k ≤ 0` returns an empty
   list.
8. `sel_roulette` and `sel_stochastic_universal_sampling` spin the
   wheel on `wvalues[0]`. A negative floor is shifted so a
   minimization weight still has a positive slice. When every slice
   is zero the draw is uniform instead of an empty list. An empty
   pool or `sel_count ≤ 0` returns `[]`.
9. `mig_ring` evicts by object identity. Two individuals with equal
   genes are no longer treated as the same slot. Emigrants are cloned
   when a replacement operator is set, so the source deme is not
   aliased into the destination. Duplicate draws take the next unused
   index instead of writing one vacancy twice.
10. `cx_messy_one_point` cuts each parent independently, so lengths
    can change. Equal-length parents no longer degenerate into a
    shared-interval two-point swap.
11. Sequence crossovers copy slices before assignment. NumPy views
    are not aliased, so a one-point or uniform swap does not destroy
    a parent. `cx_es_two_point_copy` applies the same copy to the
    strategy vector.
12. `sel_spea_2` uses the full distance row for density, with the
    self-distance set to infinity, so the $k$-th neighbour is not the
    zero pad of the upper triangle. An empty pool returns `[]`.
13. `sel_nsga_3` intercepts on the success path are $1/x + \mathrm{best}$.
    Association treats a near-zero $\mathrm{intercepts} - \mathrm{best}$
    gap as $1$ so the niche distance is not NaN. Niching stops when
    the last front is exhausted, so a $k$ larger than the pool does
    not loop forever.
14. `cx_one_point`, `cx_two_point`, `cx_ordered`, and
    `mut_shuffle_indexes` no-op when a parent is shorter than two
    genes, so a length-1 individual no longer hits an empty `randint`
    interval or a two-cut `sample` on a one-gene permutation.
15. `broadcast_param` treats `numbers.Integral` and `numbers.Real` as
    scalars, so a NumPy integer or `float32` / `float16` bound no
    longer raises `TypeError` from `len()` on a NumPy scalar.
16. `sel_lexicase` and `sel_epsilon_lexicase` accept `cases=` to
    filter on a per-generation subset of fitness indices. Defaults
    still use every case. `sample_informed_cases` builds that subset
    by farthest-first traversal of Hamming distances between case
    solve vectors, so synonymous cases are not over-sampled.
    `fitness_case_matrix` packs `fitness.values` into a dense matrix;
    optional `matrix=` and `trust_matrix=` let lexicase and informed
    down-sampling reuse one pack per generation.
17. `sel_sms_emoa` reduces a pool by non-dominated sorting, then
    removes the least hypervolume contributor on the critical front
    until the quota is met. Optional `ref_point` follows the same
    minimization-space convention as `hypervolume` and `least_contrib`.
    Works on `parents + offspring` or steady-state `parents + [child]`.
18. `sel_moead` and `SelMOEADWithMemory` pick one winner per
    decomposition weight from `uniform_reference_points`, using
    Tchebycheff or PBI scalarization with Pareto-rank-aware tie
    breaks and NSGA-II-style crowding on the fill pass.
19. `sel_age_moea_2` and `SelAGE2WithMemory` advance front by front:
    geodesic diversity on partial $F_1$, inverse Minkowski on later
    partial fronts, with Newton–Raphson curvature on the first front.
20. `mut_gaussian_bounded` applies the same $N(\mu, \sigma)$ draw as
    `mut_gaussian`, then clamps each mutated gene into `[low, up]`.
    Bounds may be scalars or per-gene sequences, matching
    `mut_polynomial_bounded`. Empty intervals are skipped.
21. `mut_de` writes a DE/rand/1/bin trial in place: $a_i + F(b_i - c_i)$
    on a binomial gene subset (`cx_prob`, at least one gene). Optional
    `low` / `up` clamp the written genes. Selection stays on the
    caller. DEAP only has this loop in examples.
22. `constraint_dominates` is Deb's feasibility-first comparison:
    feasible beats infeasible; two feasibles use ordinary Pareto;
    two infeasibles prefer the smaller constraint violation.
    `sel_nsga_2` accepts optional `feasible=` / `violation=`
    callables and ranks with that rule. Defaults stay unconstrained
    NSGA-II. Fitness values are not rewritten.
23. `sel_team` assembles `sel_count` individuals by greedy maximum
    coverage of cases solved at 0. Optional `cases=`, `matrix=`, and
    `trust_matrix=` match lexicase. The team is a sequence of pool
    members; member fitness is not rewritten. Team scoring stays on
    the caller. DEAP has no team selector.
24. `CaseExam` and `CaseExamPool` store case subsets as ranges or a
    1-D bool mask — the same shapes `case_errors` and
    `sel_lexicase(..., cases=)` consume — not as a new genome.
    `score_case_exams` ranks exams on elites by unsolved count or
    Hamming distance from the all-solved vector (solved ≡ $0$).
    `mut_case_ranges` / `mut_case_mask` vary bounds or flip mask
    runs; `guard_case_exams` blocks the empty exam and the
    all-solved collapse. `next_lexicase_cases` returns the mutated
    or guarded winner as the next `cases=` list. `informed=True`
    (the default) is a guard repair path only — it does not
    overwrite a healthy exam. Chronological splits stay on the
    caller.
25. `mig_ring` pairs source emigrants with dest vacancies by
    length, and stops claiming slots once a deme is full. Unequal
    island sizes, or `sel_random` with `k` larger than a deme, no
    longer `IndexError` or `StopIteration`. An unreplaced home
    vacancy clones the leftover emigrant so two demes do not
    share one object.
26. `sel_best` and `sel_worst` return `[]` when `sel_count <= 0`.
    A negative count is no longer a Python slice that drops
    individuals from the other end of the ranked list.
27. `mig_ring` clones an emigrant that is already present in the
    destination, so a selector that returns the same object twice
    does not write that object into two dest slots.
28. `sel_spea_2` returns `[]` when `sel_count <= 0`. A negative
    count no longer enters archive truncation and `IndexError`s
    on an empty list. Matches `sel_nsga_2` and `sel_best`.
29. `cx_partially_matched` no-ops when a parent is shorter than
    two genes. Empty permutations no longer hit `randint(0, -1)`.
30. `broadcast_param` treats a 0-d `ndarray` as a scalar, so
    `numpy.array(0.0)` bounds no longer raise `TypeError` from
    `len()` on an unsized object.
31. <a id="31-held-out-policy-fitness"></a>`policy_held_out_fitness`
    scores policy individuals only on a caller-marked `held_out`
    exam. `guard_policy_fitness_exam` refuses train exams or a
    freshly mutated exam as the objective. Train quality stays in
    `policy_exam_scores` / `policy_observe`; `record_policy_generalization_gap`
    logs train, held-out, and gap as a `generalization_gap` Logbook
    chapter ([Push GP P13](../roadmap/push_gp.md#p13-held-out-policy-fitness)).
32. `PolicyActionGuard` and `guard_policy_action` enforce hard
    caps on `apply_policy_action`: max promotes per generation,
    promote cooldown, max inner `tune` generations, minimum exam
    size, and rejection when an action would exceed remaining
    `n_evals`. Rejected actions return
    `PolicyActionResult(rejected=True)` without raising — the
    observation surface for `last_action_rejected`. DEAP has no
    policy action firewall ([Push GP P14](../roadmap/push_gp.md#p14-action-guards-and-cooldowns)).
33. `sel_novelty` ranks a pool by average distance to the `k` nearest
    archive behavior descriptors via `semantic_distance`. Fitness
    stays on `ind.fitness`; novelty is the selection key. An empty
    archive falls back to `sel_random`. `mut_iso_line` interpolates
    toward a donor elite with `t ~ Uniform(-iso, 1 + iso)` and adds
    isotropic noise; `iso_line_float`, `iso_line_int`, and
    `iso_line_bit` compose through `mut_heterogeneous` on mixed
    genomes. `ea_map_elites` registers both like any other
    `select` / `mutate`; `random_elites` stays the parent source.
    DEAP has no novelty selector or archive-aware iso+line mutator.
    See the [MAP-Elites example](../../examples/genetic_algorithms/map_elites.md).
34. `mig_fully_connected` and `mig_random` sit next to `mig_ring`.
    Fully connected selects emigrants once per source, clones them
    along each outgoing edge, and claims distinct destination
    vacancies across incoming edges; random picks one destination
    per source. `island_eval_keys` hashes each deme's `CaseExam` —
    catalog subsets via a painted mask so mask and range storage
    match — and optional matrix identity for
    `step_islands(..., eval_keys=)`. Custom graphs stay
    `mig_ring(..., mig_indices=)`. DEAP has no named topologies.
    See the [multiprocessing tutorial](../../tutorials/multiprocessing.md).
35. `sel_batch_epsilon_lexicase` shuffles active cases into batches
    of at most ``batch_size``, reduces each batch (mean squared error
    by default), and runs epsilon-lexicase on the shorter matrix.
    A fresh partition is drawn per selected individual. Optional
    ``reduction=`` overrides the batch aggregate. ``matrix=`` /
    ``trust_matrix=`` / ``cases=`` match lexicase.
    ``sel_tournament_cases`` scores individuals on a case subset
    (column mean by default), then tournaments on those scalars.
    Informed down-sampling stays on ``sample_informed_cases``;
    ``case_count=`` draws a random subset when ``cases`` is omitted.
    DEAP has no batch epsilon-lexicase or case-subset tournament.
    See the [columnar GP tutorial](../../tutorials/columnar_gp.md).
36. ``sel_epsilon_lexicase`` accepts ``mode=`` on the vectorized filter:
    ``epsilon_auto`` / ``epsilon_static`` (population MAD and elite),
    ``epsilon_semi`` (population MAD, pool elite), and
    ``epsilon_dynamic`` (pool MAD and elite). ``next_downsample_cases``
    returns the next ``cases=`` list each generation with ``mode=``
    ``random``, ``informed``, ``cohort``, or ``held_out`` (rotate
    through a caller-marked held-out exam). Chronological meaning stays
    on the caller. DEAP has no filter-pool epsilon modes or generation
    downsample schedule. See the
    [columnar GP tutorial](../../tutorials/columnar_gp.md).
37. `mut_iso_line` treats NumPy integer genes as integers. A
    `numpy.int64` gene no longer takes the float path and writes a
    non-integer value.
38. `sel_random` returns `[]` when the pool is empty. A positive
    `sel_count` no longer `IndexError`s on `rng.choice([])`.
39. `sel_double_tournament` returns `[]` when the pool is empty
    or `rounds <= 0`. It no longer `ValueError`s on `max([])`
    after `sel_random` started returning an empty draw.

[deap-321]: https://github.com/DEAP/deap/issues/321
[deap-472]: https://github.com/DEAP/deap/issues/472
[deap-527]: https://github.com/DEAP/deap/issues/527
[deap-641]: https://github.com/DEAP/deap/issues/641
[deap-655]: https://github.com/DEAP/deap/issues/655
[deap-740]: https://github.com/DEAP/deap/issues/740
[deap-755]: https://github.com/DEAP/deap/issues/755
