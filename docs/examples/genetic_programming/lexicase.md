# Informed Lexicase

Symbolic regression with one error per sample. Each generation rebuilds
an informed case subset and selects with ε-lexicase. The same script
also shows the usual `fitness_case_matrix` / `cases=` contract used by
`sel_batch_epsilon_lexicase`, `sel_tournament_cases`, and
`next_downsample_cases` — see the
[columnar tutorial](../../tutorials/columnar_gp.md) and roadmap
[item 27](../../overview/roadmap/features_21_30.md#27-batch-epsilon-lexicase-and-down-sampled-tournament)
/ [item 28](../../overview/roadmap/features_21_30.md#28-dynamic-epsilon-and-downsample-schedule).
Weighted primitives, a `call_zero` terminal, and `tree_to_infix` are on
the same script.

```python
--8<-- "examples/genetic_programming/lexicase.py"
```
