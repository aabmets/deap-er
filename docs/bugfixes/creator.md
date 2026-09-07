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
`tests/test_creator/test_overrides.py::test_copy_copy_keeps_type_and_fitness_on_array_individuals`

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
