# Records

Correctness fixes in HallOfFame, Logbook, History, and MultiStatistics.

---

## ParetoFront crashed on a missing fitness

An empty front skipped a member without `fitness` (`insert` no-ops).
Once the front held anyone, `update` compared `ind.fitness` and
raised `AttributeError`, so later valid individuals were never seen.

**Fix.** Skip an individual that has no `fitness` attribute, matching
`HallOfFame.update`.

**Validator.**
`tests/test_records/test_hall_of_fame.py::test_pareto_front_skips_individual_without_fitness`

---

## HallOfFame empty-archive bootstrap

If `population[0]` had no `fitness`, `insert` no-op'd and every later
individual was skipped. `HallOfFame(0)` fell through to `self[-1]`
and raised `IndexError`.

**Fix.** Return immediately when `maxsize == 0`. Bootstrap inserts
the first individual that has a `fitness` attribute.

**Validators.**

- `tests/test_records/test_hall_of_fame.py::test_update_skips_no_fitness_bootstrap_and_keeps_later_members`
- `tests/test_records/test_hall_of_fame.py::test_update_with_zero_maxsize_is_a_noop`

---

## HallOfFame discarded a strictly better clone

`_has_similar` ran after the fitness test and compared genotype
only, so a noisier or later-better copy of an archived genome was
dropped.

**Fix.** Skip a similar member that is equal or worse; if the
newcomer is strictly better, replace the archive member.

**Validator.**
`tests/test_records/test_hall_of_fame.py::test_update_replaces_similar_member_when_fitness_is_better`

---

## `remove` desynchronized `keys` and `items`

An out-of-range index wrap-arounded `keys` then failed on `items`,
leaving the two lists different lengths.

**Fix.** Normalize a negative index, then raise `IndexError` if it
is still out of range.

**Validator.**
`tests/test_records/test_hall_of_fame.py::test_remove_out_of_range_keeps_keys_aligned`

---

## Logbook `__delitem__` assumed chapters were row-aligned

A generation recorded without a chapter left that chapter shorter.
`del lb[i]` then popped the wrong chapter row, or raised
`IndexError` after the parent row was already gone.

**Fix.** Delete the chapter row that belongs to that parent record
(match on `gen`), not `chapter.pop(i)`.

**Validator.**
`tests/test_records/test_logbook.py::test_delitem_does_not_drop_unrelated_chapter_row`

---

## Slice delete had the same chapter desync

`_delete_slice` reused the parent's numeric index for every
chapter.

**Fix.** Walk indexes high-to-low and call `_delete_index`, which
matches chapter rows by `gen` and skips a missing one.

**Validator.**
`tests/test_records/test_logbook.py::test_delitem_slice_does_not_drop_unrelated_chapter_row`

---

## Repeated `gen` paired the wrong chapter row

Matching on `gen` alone popped the first chapter row with that
value, not the occurrence that belonged to the deleted parent
index.

**Fix.** Count parent rows with that `gen` at or after the deleted
index and pop the matching occurrence.

**Validator.**
`tests/test_records/test_logbook.py::test_delitem_pairs_matching_repeat_generation`

---

## `pop` left chapter rows behind

`__delitem__` removed the matching chapter row. `pop` only dropped
the parent entry, so `logbook.pop(i)` and `del logbook[i]` diverged:
chapters still held the deleted generation.

**Fix.** `pop` removes the chapter row that shares ``gen`` (same
occurrence rule as `__delitem__`). `__delitem__` now calls `pop`.

**Validator.**
`tests/test_records/test_logbook.py::test_pop_removes_matching_chapter_row`

---

## `pop(-1)` rewound the stream cursor

A raw negative index was compared to `buff_index`, so `pop(-1)`
looked like “before the last streamed row” and rewound the cursor.

**Fix.** Convert `index + len` before comparing to `buff_index`.

**Validator.**
`tests/test_records/test_logbook.py::test_pop_negative_index_does_not_rewind_unstreamed_cursor`

---

## Stream assumed chapters were row-aligned

A generation recorded without a chapter left that chapter shorter
than the parent. `offsets` used `len(parent)` instead of the
chapter's own row count, so the chapter header was consumed as a
data cell and later values landed on the wrong generation. `stream`
then indexed past the rendered chapter lines (`IndexError`).

**Fix.** Render each chapter through a parent-aligned view. Rows
pair by `gen` (the same occurrence rule as `__delitem__`). When
there is no `gen` and the chapter is already the same length as the
parent, pairing is positional. Missing chapter cells are blank.

**Validators.**

- `tests/test_records/test_logbook.py::test_stream_omitted_chapter_does_not_raise`
- `tests/test_records/test_logbook.py::test_str_keeps_chapter_values_on_their_generation`

---

## `to_json` / `from_json` dropped nested chapters

Only one chapter level was walked. Two-level stats (chapter of
chapters) vanished on a round-trip.

**Fix.** Recurse `chapter.chapters`.

**Validator.**
`tests/test_records/test_logbook.py::test_json_round_trip_restores_nested_chapters`

---

## `History.update` orphaned a whole batch

One individual without `history_index` made the batch's parent
edges disappear, including for members that did have an index.

**Fix.** Collect `history_index` only from individuals that have
the attribute.

**Validator.**
`tests/test_records/test_history.py::test_update_keeps_parents_when_one_individual_lacks_history_index`

---

## GridArchive NaN fitness replaced a finite elite

`add` required a valid single-objective fitness but did not require
finite values. `nan <= incumbent` is False, so a NaN candidate
replaced a finite elite and `stats.qd_score` became NaN. `inf`
did the same. An empty cell would also store a non-finite elite.

**Fix.** Reject a candidate whose first weighted objective is not
finite, the same way non-finite descriptors are rejected.

**Validator.**
`tests/test_records/test_grid_archive.py::test_add_rejects_non_finite_fitness`

---

## `MultiStatistics.compile` exhausted a generator

Each chapter iterated `data`. A one-shot iterable (generator,
`map`, `zip`) was empty for every chapter after the first.

**Fix.** `data = list(data)` once.

**Validator.**
`tests/test_records/test_statistics.py::TestStatistics::test_multi_statistics_compile_consumes_generator_once`
