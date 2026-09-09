# Island Models

Three OneMax demes. Each toolbox registers `vary` (`var_and`),
`select`, and `evaluate`. `step_islands` steps every deme, then
`mig_ring` moves emigrants. See the
[island models tutorial](../../tutorials/islands.md).

```python
--8<-- "examples/genetic_algorithms/islands.py"
```

## Island topologies

The same deme contract with `mig_fully_connected` and `mig_random`
next to `mig_ring`. See the
[island models tutorial](../../tutorials/islands.md#migration)
and roadmap
[item 35](../../overview/roadmap/features_31_40.md#35-island-topologies).

```python
--8<-- "examples/genetic_algorithms/islands_topologies.py"
```
