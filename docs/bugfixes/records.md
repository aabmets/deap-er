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
`tests/test_records/test_pareto_front.py::test_pareto_front_skips_individual_without_fitness`

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

## HallOfFame archived invalid and non-finite fitness

`update` only skipped a missing `fitness` attribute. A typical
creator individual has the attribute before evaluation, so
unevaluated members occupied hall-of-fame slots. Assigned NaN
compared as incomparable under `bisect_right`, so a NaN key
sorted to the front as if it were the best member. `ParetoFront`
kept the same invalid and all-NaN individuals on the front.

**Fix.** Skip an individual whose fitness is missing, invalid, or
non-finite, matching MAP-Elites `add`.

**Validators.**

- `tests/test_records/test_hall_of_fame.py::test_update_skips_invalid_fitness_and_keeps_later_members`
- `tests/test_records/test_hall_of_fame.py::test_update_rejects_non_finite_fitness`
- `tests/test_records/test_pareto_front.py::test_pareto_front_skips_invalid_and_non_finite_fitness`

---

## `pop` without `gen` left chapter rows behind

`pop` and `__delitem__` pair chapter rows by `gen`. A record
written without that key returned no match, so the chapter kept
every row. Stream pairing already falls back to position when
lengths match, so `str` then blanked the remaining cells.

**Fix.** When `gen` is missing and the chapter is the same length
as the parent, delete (and render) the chapter row at the same
index. A shorter chapter — a generation recorded without that
chapter — is still left alone.

**Validators.**

- `tests/test_records/test_logbook.py::test_pop_without_generation_removes_positional_chapter_row`
- `tests/test_records/test_logbook.py::test_pop_without_generation_keeps_shorter_chapter`

---

## GridArchive accepted non-finite ranges

`parse_grid_config` required `low < high` but not finite ends.
`(0, inf)` built a grid that mapped every finite descriptor to
cell 0. `(-inf, 0)` and NaN bounds crashed later in
`descriptor_to_index` with `int(nan)`.

**Fix.** Reject a range whose ends are not finite.

**Validator.**
`tests/test_records/test_grid_archive_errors.py::test_non_finite_ranges_raise`

---

## `MultiStatistics.compile` exhausted a generator

Each chapter iterated `data`. A one-shot iterable (generator,
`map`, `zip`) was empty for every chapter after the first.

**Fix.** `data = list(data)` once.

**Validator.**
`tests/test_records/test_statistics.py::TestStatistics::test_multi_statistics_compile_consumes_generator_once`

---

## `HallOfFame.insert` archived invalid and non-finite fitness

`update` already skips missing, invalid, and non-finite fitness.
`insert` only required a `fitness` attribute, so a direct insert
or `from_json` restore could archive `wvalues=()` or NaN. Those
keys then sorted as if they were the best member.

**Fix.** `insert` uses the same comparable-fitness guard as
`update`. `ParetoFront.insert` and JSON restore go through that
path.

**Validators.**

- `tests/test_records/test_hall_of_fame_insert.py::test_insert_skips_invalid_and_non_finite_fitness`
- `tests/test_records/test_hall_of_fame_insert.py::test_pareto_insert_skips_invalid_and_non_finite_fitness`
- `tests/test_records/test_hall_of_fame_insert.py::test_from_json_does_not_restore_non_finite_fitness`

---

## `clear` left chapter rows and the stream cursor

`pop` and `__delitem__` keep chapters and `buff_index` aligned.
`list.clear` does not go through those methods. After `clear()`,
chapters still held every generation. After a stream, `clear()`,
and a new `record`, `stream` started at the stale cursor and
returned no rows.

**Fix.** `clear` deletes every parent row through `__delitem__`
(the same chapter and cursor rules as `del logbook[:]`).

**Validators.**

- `tests/test_records/test_logbook_edges.py::test_clear_removes_chapter_rows`
- `tests/test_records/test_logbook_edges.py::test_clear_after_stream_does_not_drop_new_rows`
