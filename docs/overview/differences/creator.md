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
5. `creator.create_type` returns the class and is idempotent: an
   equal definition under an existing name returns the existing class
   without a warning, so live instances, `isinstance` checks, and
   pickles keep working. A name that clashes with the module's own
   attributes (`array`, `warnings`, `create_type`, …) raises
   `ValueError` instead of overwriting the import that `create_type`
   itself needs.
