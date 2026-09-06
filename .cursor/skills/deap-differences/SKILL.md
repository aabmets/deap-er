---
name: deap-differences
description: >-
  Keeps the Relative-to-DEAP scoreboard and differences inventory in sync.
  Use when fixing a DEAP bug, implementing a still-open DEAP issue, adding a
  capability DEAP does not have, or editing README.md / docs/overview/differences.md
  scoreboard counts.
---

# DEAP differences scoreboard

The three-line scoreboard lives in two places and must stay identical
except for hrefs:

| File | Role |
|:-----|:-----|
| `docs/overview/differences.md` | Inventory (source of truth) + scoreboard |
| `README.md` (`## Relative to DEAP`) | Same three counts; Pages URLs |

This update is **in scope** for a DEAP bug, DEAP issue, or new
capability. Do not ask first. Do not expand into tutorials, reference,
or other docs unless those are already in scope. Follow
[`mkdocs-docs`](../mkdocs-docs/SKILL.md) for Material / math syntax.

## When this applies

**Do** after the code change lands:

- Implementing a still-open [DEAP](https://github.com/DEAP/deap) issue
- Fixing a crash, NaN, wrong result, or inherited defect vs DEAP
- Adding an operator, flag, statistic, backend, or algorithm kwarg
  DEAP does not have
- Reverting or narrowing such an item

**Skip** (mention in the reply if useful):

- Rewrite / rename / typing / snake_case only
- Tests, refactors, Sonar cleanup, performance-only (that is
  `docs/overview/performance.md`)
- Docs-only work that does not change the inventory

## Workflow

1. Add or edit **one numbered item** in the matching section of
   `docs/overview/differences.md` (not under `## Rewrite and public API`).
   Match the tone and density of neighboring items.
2. If it implements a DEAP issue, cite it (`[deap-N]`) and add the
   footnote next to the existing `[deap-*]` list. Confirm the issue
   exists on `DEAP/deap` before linking.
3. **Recount from scratch.** Do not increment the old numbers by
   habit — a split, merge, or reclassification can move two counts.
4. Write the same three bullets into **both** files. Only the links
   differ (relative in docs, `https://aabmets.github.io/deap-er/…` in
   the README).
5. Leave the README closer (typed / snake_case / Apache / moocore /
   performance) alone unless a count line must mention a new headline
   capability.

## How to count

Ignore `## Rewrite and public API`. Walk every later numbered item.

| Bucket | What increments it |
|:-------|:-------------------|
| **Issues** | Unique `[deap-N]` footnotes. One issue, one count, even if two sentences cite it. |
| **Bugs** | Crash, exception, NaN/complex, wrong numeric result vs a published formula, aliased parent, skipped strictly-better individual, desynchronized keys, truncated checkpoint, exhausted generator, orphaned history, empty return when individuals exist, IndexError on valid input. |
| **Features** | New operator, optional flag/path, statistic, serialization, algorithm kwarg, GP kit/backend, or accepting a previously rejected valid `k`. Stream/header cosmetics are features, not bugs. |

Rules:

- Each numbered item is **bug XOR feature**, never both.
- Several sub-defects in one item (HARM, Chuang F3, Moving Peaks)
  still count as **one** bug.
- Several DEAP issue links in one **feature** item (the `ea_*`
  `log_time` / `logger` / `fronts` bullet) increment **features** by
  the number of those issues.
- A DEAP issue that is a crash/wrong-result ticket counts as a **bug**
  and as an **issue**. A feature-request ticket counts as a **feature**
  and as an **issue**. The 18 is not a third pile of work.
- `generate()` closing a leaf-only type is a **feature**.
- `sel_tournament_dcd` accepting any valid `k` is a **feature**.
- Empty `Logbook` printing its header is a **feature**.

Baseline that these rules reproduce (do not hard-code; re-derive):

`18` issues · `27` bugs · `21` features

## Scoreboard copy

Docs (`differences.md` already has relative links):

```markdown
- **N** still-open [DEAP](https://github.com/DEAP/deap) issues
  implemented (some older than a decade)
- **N** correctness bugs fixed — operators, GP, CMA, records,
  checkpoints, and published benchmarks
- **N** capabilities DEAP does not have, including boxed CMA,
  mixed-gene mutation, logbook JSON, and
  [columnar GP](../tutorials/columnar_gp.md)
```

README uses the same sentences with

`https://aabmets.github.io/deap-er/overview/differences/` and
`https://aabmets.github.io/deap-er/tutorials/columnar_gp/`.

Keep the “including …” examples unless the new capability is more
notable than boxed CMA, mixed-gene mutation, logbook JSON, or
columnar GP.

## Done when

- [ ] Inventory item is in the right section, numbered, and (if
      applicable) linked
- [ ] All three counts re-derived from the full inventory
- [ ] README and `differences.md` show the same three numbers and
      the same bullet wording (hrefs excepted)
