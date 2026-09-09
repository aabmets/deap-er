# Batch and Dynamic Lexicase

Symbolic regression with three selection modes in one run:
`sel_batch_epsilon_lexicase`, down-sampled `sel_tournament_cases`, and
`sel_epsilon_lexicase` with `mode="epsilon_dynamic"` plus
`next_downsample_cases`. See the
[columnar tutorial](../../tutorials/columnar_gp.md) and roadmap
[item 27](../../overview/roadmap/features_21_30.md#27-batch-epsilon-lexicase-and-down-sampled-tournament)
/ [item 28](../../overview/roadmap/features_21_30.md#28-dynamic-epsilon-and-downsample-schedule).

```python
--8<-- "examples/genetic_programming/lexicase_batch.py"
```
