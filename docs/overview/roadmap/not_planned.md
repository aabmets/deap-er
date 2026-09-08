# Not planned

These ideas stay off the library surface. They are listed so the
numbered backlog is not read as “everything in the GP literature.”

See the [overview](index.md) for what *is* planned, and
[Under consideration](under_consideration.md) for ideas that may
still join that table.

| Idea | Why not |
|:-----|:--------|
| PushGP, Cartesian GP, or linear GP as a second public genome | Prefix trees plus the tape already linearize a program. A second representation needs a caller who cannot use trees or columnar arrays. A *private* Push policy that only sees summaries is a different track: [Push GP](push_gp.md). |
| Transformer or LLM mutation | Heavy optional dependencies, unstable operators, and they do not compose with the tape. A recipe in a notebook is enough. |
| A catalog of named domain indicators as primitives | Composable windows and user opcodes. Named catalogs age badly. |
| Built-in domain fitness functions | They need application state the library does not own. |
| Learned quality-diversity / meta-BBO | A research paper, not a toolbox function. [Item 22](features_21_30.md#22-semantic-search-space) is a semantic descriptor and nearest-neighbor, not a trained QD model. |
| Interactive / human-in-the-loop evolution | A notebook over `archive` + `tree_to_infix`. Machine-checkable pressure is items 22–23 and 26, not clicks. |
| Switching persistence off dill, or replacing `creator` with dataclasses | Forbidden by the project contract. |
| An in-tree C / Cython rewrite of operators, CMA, or selection | Not the bottleneck; those modules must stay readable. Revisit an *optional* tape backend only after items 3, 4, 9, and [32](features_31_40.md#32-population-tape-cse) are in and a profile still points at `interpret`. |
