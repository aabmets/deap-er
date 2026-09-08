# NSGA-III

DTLZ2 with three objectives. `SelNSGA3WithMemory` keeps the ideal,
nadir, and extreme points across generations. Reference points come
from `uniform_reference_points`. See the
[operators tutorial](../../tutorials/operators_and_algorithms.md) and
the [Operators](../../reference/operators.md) reference.

```python
--8<-- "examples/genetic_algorithms/nsga3.py"
```
