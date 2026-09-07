# Boxed CMA

Standard CMA with box bounds and `bound_mode="resample"`. Out-of-box
draws are rejected and redrawn; after `resample_limit` failures the
sample is clipped.

```python
--8<-- "examples/evolution_strategies/cma_boxed.py"
```
