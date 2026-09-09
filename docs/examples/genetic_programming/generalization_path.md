# Generalization Path

`case_generalization_recipe` keeps lexicase on train cases only, then
`evaluate_case_halving` spends a case budget before full scoring. See the
[columnar tutorial](../../tutorials/columnar_gp.md#generalization-path)
and roadmap
[item 41](../../overview/roadmap/features_41_50.md#41-case-structured-generalization-path).

```python
--8<-- "examples/genetic_programming/generalization_path.py"
```
