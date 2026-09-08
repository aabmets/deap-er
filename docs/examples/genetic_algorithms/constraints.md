# Constrained Search

A two-objective point with a spherical feasible region.
`DeltaPenalty` wraps `evaluate`; `sel_nsga_2` ranks with Deb
constraint-dominance (`feasible=` / `violation=`). See the
[constraints tutorial](../../tutorials/constraints.md).

```python
--8<-- "examples/genetic_algorithms/constraints.py"
```
