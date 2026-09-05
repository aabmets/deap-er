# Columnar Programs

Rediscovers a causal program over a table of numeric columns. The search
uses the vectorized primitive kit and the causal window primitives, and
the fitness function masks the `nan` warmup that those windows produce.

See the [Columnar Programs tutorial](../../tutorials/columnar_gp.md) for a
walkthrough of the pieces used here.

```python
--8<-- "examples/genetic_programming/columnar_gp.py"
```
