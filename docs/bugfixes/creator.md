# Creator

Correctness fixes in `creator.create` and the numpy / `array` overrides.

---

## `copy.copy` drops `fitness` on numpy and `array` individuals

`_NumpyOverride` and `_ArrayOverride` implemented `__deepcopy__` but
not `__copy__`. `copy.copy` on a numpy individual kept the subclass
and dropped `__dict__` (`fitness` gone). On an `array` individual it
returned a plain `array.array`.

**Fix.** Both overrides implement `__copy__`: shallow buffer copy plus
a rebound `fitness`, matching `__deepcopy__`.

**Validator.**
`tests/test_overrides.py::test_copy_copy_keeps_type_and_fitness_on_array_individuals`

---

## `create` ignores an `array.array` instance typecode

`create(..., array.array("d"), ...)` is supposed to unwrap the instance
to its class and keep typecode `"d"`. The unwrap guard used
`hasattr(base, "__module__")`, which is `True` on `array` instances, so
the instance was replaced with `_ArrayOverride` and its default
typecode `"b"`. Float genomes then raised `TypeError`.

**Fix.** An `array.array` instance is unwrapped to the class and
`base.typecode` is copied onto the created type.

**Validator.**
`tests/test_creator/test_creator.py::TestCreatorBuiltinsArray::test_array_instance_keeps_typecode`

---

## `Fitness.values` crashed on a 0-d NumPy array

`numpy.array(1.0)` is `Iterable`, so the setter called
`tuple(values)`. Iterating a 0-d array raises `TypeError`.
NumPy scalars (`float64`, `int64`) and a 1-d array already
worked. Bounded operators already treat a 0-d bound as a scalar.

**Fix.** A 0-d array is unwrapped with `.item()` before the
length check.

**Validator.**
`tests/test_fitness.py::TestFitness::test_numpy_0d_array_is_accepted`

---

## `Fitness.dominates` crashed on invalid fitness

An unevaluated fitness has empty `wvalues`. The two- and three-objective
fast paths unpacked that empty tuple, and the other lengths indexed it.
`valid.dominates(invalid)` raised `ValueError` or `IndexError` instead of
reporting that dominance does not hold.

**Fix.** Return `False` when either side has no weighted values or the
compared lengths differ.

**Validator.**
`tests/test_fitness.py::TestFitness::test_dominates_invalid_fitness_is_false`
