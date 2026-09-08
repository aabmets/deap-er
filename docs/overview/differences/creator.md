# Creator

1. `creator.create` keeps the `typecode` of an `array.array`
   instance base. It no longer forces `"b"`.
2. `Fitness.dominates` returns `False` when the other fitness is
   invalid or the compared objective counts differ. An unevaluated
   opponent no longer raises `ValueError` or `IndexError`.
3. `copy.copy` on a numpy or `array.array` individual keeps the
   created type and rebinds `fitness`. The overrides no longer
   drop `__dict__` or return a plain `array.array`.
4. `Fitness.values` accepts a 0-d `ndarray`. `numpy.array(1.0)` no
   longer raises `TypeError` from iterating an unsized array.
