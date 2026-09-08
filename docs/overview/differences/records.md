# Records, statistics, and algorithms

1. `MultiStatistics.register(..., chapters=)` can target a subset of
   chapters, so fitness min/max and a size statistic need not share
   one function ([DEAP#720][deap-720]).
2. `MultiStatistics.compile` materializes the input once. A
   generator, `map`, or `zip` is no longer exhausted by the first
   chapter.
3. An empty `Logbook` with a `header` [prints that header][deap-694]
   from `stream()` instead of the word “empty”.
4. `Logbook.to_json` / `from_json` [round-trip][deap-121] entries,
   nested chapters, and the header. NumPy scalars become Python
   numbers.
5. `duplicate_count` is a [variety statistic][deap-350]: population
   length minus distinct keys. Hashable keys scan in linear time;
   unhashable but sortable keys use an adjacent-run count after sort.
6. The four `ea_*` algorithms accept `log_time` ([per-generation
   wall time][deap-426]), `logger` ([instead of only
   `print`][deap-750]), and `fronts` (append a new `ParetoFront` of
   that generation's survivors, [not the cumulative
   hall of fame][deap-735]).
7. `HallOfFame.update` inserts into an empty archive, no-ops at
   `maxsize=0`, and replaces a similar member when the new individual
   is strictly better.
8. `ParetoFront.update` skips an individual that has no `fitness`
   instead of raising `AttributeError` after the front already holds
   a member.
9. `HallOfFame.remove` raises `IndexError` on an out-of-range index
   instead of desynchronizing `keys` and `items`.
10. `Logbook.pop` normalizes a negative index before comparing it to
    the stream cursor.
11. `Logbook.pop` removes the chapter row that shares that
    generation, so `pop(i)` and `del logbook[i]` stay aligned.
12. `Logbook.__delitem__` removes the chapter row that shares the
    same generation — including a later occurrence of a repeated
    `gen` and every index in a slice — not the same list index.
13. `Logbook.stream` and `str` pair chapter cells by `gen`. A
    generation recorded without a chapter no longer shifts later
    values onto the wrong row or IndexErrors once the stream cursor
    is past the shorter chapter.
14. `History.update` records every member of a batch. A single
    individual without `history_index` no longer orphans the rest.
15. `GridArchive` tessellates behavior descriptors into a MAP-Elites
    grid: `add` keeps the best individual per cell, `random_elites`
    samples parent copies, and `stats` reports coverage and
    `qd_score`. `ea_map_elites` drives evaluate → archive → `var_or`
    and logs archive metrics each generation. Fitness stays on
    `ind.fitness`; behavior measurement stays on the caller.
16. `var_or` mates two clones of the only parent when the pool has a
    single individual, so $(1,\lambda)$ / $(1+\lambda)$ with
    `cx_prob > 0` no longer raises `ValueError` on `sample(..., 2)`.
    `ea_mu_comma_lambda` with `survivors=1` can run past generation
    one.
17. `GridArchive.add` rejects a non-finite first weighted objective.
    NaN or infinity no longer replaces a finite elite or occupies an
    empty cell.
18. `CvtArchive` and `UnstructuredArchive` sit next to `GridArchive`.
    CVT keeps one elite per k-means / caller centroid; unstructured
    adds a point that is far enough from every member or replaces
    the nearest neighbor. `cvt_centroids` builds the CVT tessellation
    from a behavior sample. `ea_map_elites` accepts either archive.
19. `step_islands` runs one evaluate → vary → select step on each
    deme, then an optional `migrate` (typically `mig_ring`). Each deme
    has its own toolbox, so selection pressure can differ while the
    topology stays a ring. Migrants keep fitness when `eval_keys`
    match; distinct keys invalidate arrivals. Append-only columnar
    evaluation may full-rescore with `interpret_tapes` after
    `vstack`, or use `tape_lookback` / `suffix_rescore` for a
    legal dirty suffix — DEAP has neither.
20. `HallOfFame.update` and `ParetoFront.update` skip an individual
    whose fitness is missing, invalid, or non-finite. An unevaluated
    creator individual no longer occupies a slot. NaN no longer
    sorts to the front of `keys` as if it were the best member.
21. `Logbook.pop` pairs a row without `gen` positionally when the
    chapter is the same length as the parent. Deleting that row no
    longer leaves chapter values behind or blanks the remaining
    cells.
22. `GridArchive` rejects a range whose ends are not finite.
    `(0, inf)` no longer maps every descriptor to cell 0, and
    `(-inf, high)` no longer crashes `descriptor_to_index` with
    `int(nan)`.
23. `ea_generate_update_restarts` keeps the last evaluated population
    when `generate` returns empty. The empty batch is still a stop
    signal; it no longer overwrites a finished run with `[]`.
24. `Logbook.clear` deletes every parent row through `__delitem__`.
    Chapters and the stream cursor stay aligned, matching
    `del logbook[:]`. `list.clear` no longer leaves chapter
    generations behind or drops later `stream` rows.
25. `ea_map_elites` skips `stats.compile` when the seed or offspring
    list is empty. A pre-filled archive with an empty `initial`, or
    generation zero with no individuals, no longer raises
    `ValueError` from `max` / `numpy.max` on an empty reduction.
    Archive metrics still record. The same guard is in
    `record_generation` for the other `ea_*` drivers.
26. `var_or` rejects each of `cx_prob` and `mut_prob` outside
    `[0, 1]`. A negative component whose sum still sits in
    `[0, 1]`, or a NaN, no longer produces offspring.
27. `ea_generate_update` keeps the last evaluated population
    when `generate` returns empty. The empty batch is still a
    stop signal and does not call `update([])`.
28. `n_evals=` is an optional evaluation-budget stop on
    `ea_simple`, `ea_mu_plus_lambda`, `ea_mu_comma_lambda`, and
    `ea_map_elites`. Generations stay the default. `EvalCache`
    wraps `evaluate` / `evaluate_batch` by expression text (or
    a caller key) plus matrix identity and row count. A hit
    does not call `evaluate` again; `n_evals` / `nevals` still
    count the fitness assignment. `promote_subtree` and
    `tune_ephemerals` drop matching fitness-cache keys when they
    invalidate the compile cache. DEAP's `ea_*` drivers stop on
    generations only.
29. `apply_policy_action` maps a discrete policy token onto
    existing toolbox callables only: `next_lexicase_cases`,
    `tune_ephemerals`, `promote_subtree`, `evaluate_invalid`,
    `interpret_tapes`, and `step_islands`. Skip tokens are
    intentional no-ops; unknown tokens and missing required
    kwargs are rejected without raising. Fitness assignment and
    rescore ownership stay on the caller — not a second `ea_*`
    driver. DEAP has no policy action schema.
30. `HallOfFame.to_json` / `from_json` [round-trip][deap-25]
    `maxsize` and archive members as ``genes`` plus ``fitness``
    values. `Checkpoint(..., hof_ind_cls=)` stores ``hof`` as
    JSON instead of dill and rebuilds it on load. DEAP has no
    text serialization for the hall of fame.

[deap-25]: https://github.com/DEAP/deap/issues/25
[deap-121]: https://github.com/DEAP/deap/issues/121
[deap-350]: https://github.com/DEAP/deap/issues/350
[deap-426]: https://github.com/DEAP/deap/issues/426
[deap-694]: https://github.com/DEAP/deap/issues/694
[deap-720]: https://github.com/DEAP/deap/issues/720
[deap-735]: https://github.com/DEAP/deap/issues/735
[deap-750]: https://github.com/DEAP/deap/issues/750
