# Evaluation Budget

`EvalCache` wraps `evaluate` so identical bit-strings are not scored
twice. `n_evals=` stops `ea_simple` after the generation that meets
the budget. See
[Operators and Algorithms](../../tutorials/operators_and_algorithms.md).

```python
--8<-- "examples/genetic_algorithms/eval_budget.py"
```
