# Genetic programming

Correctness fixes in crossover, mutation, HARM, typed primitive sets,
and tree assignment.

---

## `cx_semantic` second child used the rewritten first parent

`create_ind` aliased `ind`. After building $o_1$ from $p_1$, the
second call received $o_1$ as the “other parent”, so $o_2$ was
$r\cdot p_2 + (1-r)\cdot o_1$ and roughly twice as long.

**Fix.** Snapshot both parents (`parent1 = list(ind1)`,
`parent2 = list(ind2)`) before rewrite.

**Validator.**
`tests/test_gp/test_semantic.py::test_semantic_crossover_second_child_uses_original_first_parent`

---

## `cx_one_point` dropped types when `ind1.root.ret is object`

The untyped fast path fired whenever the first parent's root returned
`object`, which is a legal typed return. Incompatible subtrees were
swapped; the same pair in reverse order stayed type-safe.

**Fix.** Always `_collect_indices`. A typed tree whose root is
`object` still only swaps compatible subtrees.

**Validator.**
`tests/test_gp/test_crossover.py::test_typed_object_root_does_not_swap_incompatible_subtrees`

---

## `static_limit` aliased both oversized children

`rng.choice(keep_inds)` stored a reference. When both children
exceeded `max_value`, about half of the pairs were one object:
mutating or evaluating child 0 rewrote child 1.

**Fix.** `new_inds[i] = deepcopy(rng.choice(keep_inds))`.

**Validator.**
`tests/test_gp/test_tools.py::test_static_limit_does_not_alias_two_oversized_children`

---

## HARM half-life scaled with size, not cutoff

Documented $\tau = \alpha\cdot x_c + \beta$ (a per-generation
constant). `_target_prob` used $\tau = \alpha\cdot x + \beta$, so the
tail decayed like $1/x$ instead of exponentially.

**Fix.** Constant half-life $\tau = \text{cutoff}\cdot\alpha + \beta$.

**Validator.**
`tests/test_gp/test_harm.py::test_target_prob_half_life_scales_with_cutoff_not_size`

---

## HARM cutoff used unevaluated model individuals

`natural_pop` is never evaluated. Sorting it by fitness ranked
“just bred” (invalid) vs clones, and the slice index used
`pop_len` instead of `len(natural_pop)`. A small `nb_model` raised
`ValueError: min() iterable argument is empty`.

**Fix.** `_cutoff_size` uses valid-fitness individuals, indexes
`int(len(source) * rho - 1)`, and returns `min_cutoff` if the slice
is empty. `harm` passes the already-evaluated `population`.

**Validator.**
`tests/test_gp/test_harm.py::test_harm_cutoff_uses_evaluated_population_and_survives_small_model`

---

## `rename_arguments` could shadow a primitive or duplicate a parameter

`add_primitive` and `add_terminal` already reject a name that matches
an argument. The inverse was open: `rename_arguments(ARG0="add")`
after registering `add` overwrote the primitive in `mapping`, and
`compile_tree` emitted `lambda add: add(...)`. Renaming two inputs
to the same name produced `lambda x, x: ...` and raised
`SyntaxError`.

**Fix.** Reject a new name that is already an argument, or already
present in `mapping` or `context`.

**Validators.**

- `tests/test_programming/test_primitives/test_primitive_set_typed.py::test_rename_arguments_rejects_name_that_matches_a_primitive`
- `tests/test_programming/test_primitives/test_primitive_set_typed.py::test_rename_arguments_rejects_name_that_matches_a_terminal`
- `tests/test_programming/test_primitives/test_primitive_set_typed.py::test_rename_arguments_rejects_duplicate_argument_name`
- `tests/test_programming/test_primitives/test_primitive_set_typed.py::test_rename_arguments_rejects_column_name_that_matches_a_primitive`

---

## `mut_insert` crashed when a sibling type had no terminals

A legal typed set can have an intermediate type with primitives but
no terminals. `mut_insert` still tried to draw a terminal and raised.

**Fix.** Return the individual unchanged, same no-op as when no
wrapping primitive exists.

**Validator.**
`tests/test_programming/test_mutation.py::test_mut_insert_skips_when_sibling_type_has_no_terminals`

---

## Primitive names that match an argument compiled as the column

`add_primitive("price", ...)` on a set whose argument is `price`
stringified as `price(...)` and `compile` bound the column, not the
function.

**Fix.** `add_primitive` and `add_terminal` reject
`name in self.arguments` with the same `ValueError` as
`reject_shadowed`.

**Validators.**

- `tests/test_programming/test_primitives/test_primitive_set_typed.py::test_add_primitive_rejects_name_that_matches_an_argument`
- `tests/test_programming/test_primitives/test_primitive_set_typed.py::test_add_terminal_rejects_name_that_matches_an_argument`

---

## `PrimitiveTree` slice assignment crashed when the start was omitted

`tree[:]` and `tree[:n]` build a slice whose `start` is `None`.
`__setitem__` compared `key.start >= len(self)` and raised
`TypeError` instead of replacing a complete tree. Same comparison
as upstream DEAP.

**Fix.** Treat a missing start as `0`, same as list assignment.

**Validators.**

- `tests/test_programming/test_primitives/test_primitive_tree.py::test_setitem_open_start_slice_replaces_whole_tree`
- `tests/test_programming/test_primitives/test_primitive_tree.py::test_setitem_stop_only_slice_replaces_whole_tree`

---

## `from_string` rejected integer `Window` leaves

`Window` is a type tag whose runtime value is `int`. `str(tree)`
writes a window as `3`, but `from_string` required
`issubclass(int, Window)`. Every windowed tree failed to
round-trip. `compile_tree` of that text on the opcode backend
calls `from_string` while lowering, so a cold cache raised
`TypeError` even though the Python backend `eval`s the same
string. A warm cache keyed by the text could hide the miss.

**Fix.** Accept a Python `int` literal in a `Window` slot. `bool`
and non-integers stay rejected.

**Validators.**

- `tests/test_programming/test_columnar.py::test_from_string_accepts_a_window_integer_leaf`
- `tests/test_programming/test_columnar.py::test_stringified_window_tree_round_trips`
- `tests/test_programming/test_columnar.py::test_opcode_backend_compiles_a_stringified_window_tree`
- `tests/test_programming/test_columnar.py::test_from_string_rejects_a_non_integer_window_literal`

---

## `from_string` compiled leftover tokens as the program

`from_string` never checked that the token stream was exactly one
complete tree. Extra tokens after a finished expression became a
second root; a truncated call left unused argument types on the
deque. `__str__` and the Python `compile_tree` path then returned
the last completed fragment.

`add(ARG0, 2, 3)` parsed as `[add, ARG0, 2, 3]`, stringified as
`3`, and compiled to the constant $3$ instead of $ARG0+2$.
`add(ARG0)` stringified as `ARG0` and compiled as the identity.
Same tokenizer as upstream DEAP.

**Fix.** Planned: reject a token that arrives after the root is
complete, and reject a stream that still owes argument types.

**Validators.**

- `tests/test_programming/test_primitives/test_primitive_tree.py::test_from_string_rejects_an_extra_argument`
- `tests/test_programming/test_primitives/test_primitive_tree.py::test_from_string_rejects_a_trailing_literal`
- `tests/test_programming/test_primitives/test_primitive_tree.py::test_from_string_rejects_an_incomplete_call`
