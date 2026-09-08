# Strongly Typed Classification

A tiny in-script float/bool table (Spambase reworked; no network).
`PrimitiveSetTyped` enforces that comparisons return `bool` and
arithmetic stays on `float`. See the
[genetic programming tutorial](../../tutorials/genetic_programming.md).

```python
--8<-- "examples/genetic_programming/typed_classify.py"
```
