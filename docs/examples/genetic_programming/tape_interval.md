# Interval Analysis on Tapes

Column bounds from `bounds_from_matrix`, static certificates with
`tape_interval` / `tape_flags`, and skipping dead programs through
`evaluate_columnar(..., static_filter=True)`. The opcode backend also
shares subgraphs across a population batch (population tape CSE inside
`interpret_tapes`). See roadmap
[item 32](../../overview/roadmap/features_31_40.md#32-population-tape-cse)
/ [item 37](../../overview/roadmap/features_31_40.md#37-interval-analysis-on-tapes).

```python
--8<-- "examples/genetic_programming/tape_interval.py"
```
