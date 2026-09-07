# Mixed Encoding

A widget with a flag, an integer count, a material choice, and two boxed
reals. Per-gene `mut_heterogeneous` mutators keep each type in range.
`cx_heterogeneous` mates the discrete prefix with `cx_uniform` and the
boxed tail with `cx_blend_bounded`.

```python
--8<-- "examples/genetic_algorithms/mixed_encoding.py"
```
