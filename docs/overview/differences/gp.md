# Genetic programming

Prefix-tree GP (loosely typed, strongly typed, ADFs) is still there.
The following is extra.

1. `generate()` closes a type that has terminals but no primitives.
   A leaf-only type — a rolling window length is the usual case —
   can appear in a strongly typed tree.
2. `add_primitive(..., weight=)` implements
   [weighted primitive sampling][deap-383]. Equal weights keep the
   previous RNG stream. The same weights apply to node replacement
   and insert mutation. Terminals stay uniform.
3. A zero-arity callable terminal can [format as `name()`][deap-644]
   when `add_terminal(..., call_zero=True)`, so the default `eval`
   compile path calls it instead of looking up the function object.
   Action terminals (Santa Fe ant) stay uncalled names by default.
4. `tree_to_infix` is an [infix pretty-printer][deap-24] for logs and
   papers. It is display-only.
5. `make_column_pset(names)` builds a strongly typed set with one
   `Array` input per column. The type tags are `Array` (1D
   `float64`), `Mask` (1D `bool`), and `Window` (`int`). Argument
   order is column order.
6. `add_numpy_primitives` registers a vectorized kit: arithmetic,
   protected `vdiv` / `vlog` / `vsqrt`, comparisons that produce a
   `Mask`, mask logic, and `vwhere`. Protected ops only replace a
   non-finite result that the operation itself fabricated; an input
   `nan` comes back out as `nan`.
7. `add_window_primitives` registers causal `delay`, `diff`,
   `rolling_{sum,mean,std,min,max}`, and `ema`. The window is
   `[t-n+1, t]`; samples without enough history are `nan`. Look-ahead
   is forbidden. `add_window_ephemeral` samples inclusive integer
   lengths.
8. `compile_tree` caches the default `eval` backend by expression
   text and context identity. `backend="opcode"` lowers the tree to
   a postfix tape and runs a NumPy stack machine. `backend="numba"`
   (the `deap-er[numba]` extra) runs the same tape in one
   process-wide compiled interpreter. Custom kernels bind at or above
   `USER_BASE` and pass one dispatcher; they are not a second
   interpreter.
9. Algorithms call `toolbox.evaluate_batch(invalids)` when that
   operator is registered, otherwise `toolbox.map(toolbox.evaluate,
   invalids)`. `ea_generate_update` and `ea_generate_update_restarts`
   go through the same `evaluate_invalid` path. A generation can be
   scored against one shared matrix without changing `map`'s
   contract.
10. `tools.clone_individual` shallow-copies a list/array individual
    and deepcopies only the fitness. GP toolboxes should register it;
    the default Toolbox clone remains `deepcopy`.
11. `cx_semantic` builds each child from a snapshot of the original
    parents. The second child is no longer derived from the already
    mutated first child.
12. `cx_one_point` always groups nodes by return type. An `object`
    root on the first parent no longer disables strongly typed
    matching.
13. `static_limit` replaces an oversized offspring with a
    `clone_individual` copy of a parent, so the two offspring slots
    never share one parent object.
14. HARM places the size cutoff on evaluated individuals only, scales
    the half-life by the cutoff (not by each individual's size), and
    does not crash on an empty candidate slice.
15. `add_primitive` and `add_terminal` reject a name that matches a
    primitive-set argument, so a compiled lambda parameter cannot
    shadow the symbol. `rename_arguments` rejects the inverse: a new
    name that is already an argument, primitive, or terminal.
16. `mut_insert` leaves the tree unchanged when a sibling type has
    no terminals, instead of raising `IndexError`.
17. `add_pair_window_primitives` registers causal `rolling_corr`,
    `rolling_cov`, and `rolling_beta` over two `Array` arguments and
    a `Window`. Moments use the population divisor; beta is the OLS
    slope of the first series on the second. Python, opcode, and
    Numba paths agree.
18. `add_ts_primitives` registers causal `ts_rank`, `ts_argmax`, and
    `ts_argmin`. Rank is the average rank of the current sample
    scaled to $[0, 1]$; a window of 1 is `nan`. Arg-extremum is how
    many samples ago the extreme occurred (`0` is now); a tie keeps
    the most recent. Python, opcode, and Numba paths agree.
19. `interpret_tapes` scores many tapes against one packed
    `(rows, columns)` matrix and returns `(n_individuals, n_rows)`.
    The opcode path unpacks columns once. The Numba path is a
    compiled loop; `parallel=True` gives each thread its own
    workspace. Unique programs are compiled once by `str(tree)`
    and lowered from the tree object. `tape_lookback` returns
    the program's causal bound; `suffix_rescore` writes a
    dirty suffix onto a cached prefix so the series matches
    that full-matrix oracle.
20. `SlimTree` stores a GP head plus semantic delta blocks. `mut_slim`,
    `mut_slim_inflate`, and `mut_slim_deflate` append or remove deltas
    without re-wrapping the whole tree; `cx_slim_donor` swaps a donor
    block size-preservingly. `compile_slim_tree` evaluates
    $\mathrm{head} + \sum \delta_i$.
21. `PrimitiveTree` slice assignment treats a missing start as `0`.
    `tree[:]` and `tree[:n]` no longer raise `TypeError` when the
    replacement is a complete tree.
22. `tune_ephemerals` extracts ephemeral floats and `Window` ints
    in documented prefix order (`SlimTree`: head, then deltas),
    runs a short boxed `Strategy` / `StrategySeparable`
    `generate` / `update` loop, writes repaired values back, and
    invalidates fitness plus the compile-cache entry for the old
    expression.
23. `semantic_moments`, `semantic_solve_bits`, and `semantic_project`
    turn an `interpret_tapes` `(n_individuals, n_rows)` pack into a
    behavior descriptor (per-row moments, lexicase solve bits, or a
    caller PCA / random basis). `semantic_nearest` looks up cosine
    or Euclidean neighbors on the finite / `valid=` mask.
    `SemanticSurrogate` is a last-generation linear or nearest-neighbor
    stand-in. Fitness stays on `ind.fitness`; the archive still ranks
    a cell by fitness.
24. `promote_subtree` lifts a complete typed subtree into the same
    `PrimitiveSetTyped` as a generated primitive (`promo0`, …).
    Later `generate` / mutation can sample that name. The library
    is capped; the least-used promoted name is evicted, not a
    built-in. Columnar sets bind at `USER_BASE` and `lower_tree`
    expands the body so tapes stay on builtin opcodes. `add_adf`
    remains the static path.
25. `PrimitiveTree.from_string` accepts an `int` literal in a
    `Window` slot. `str(tree)` writes window lengths as integers;
    the opcode backend no longer `TypeError`s when compiling that
    text, and a stringified windowed tree round-trips.
26. `PrimitiveTree.from_string` rejects extra tokens and incomplete
    calls. `add(ARG0, 2, 3)` and `add(ARG0)` no longer stringify
    as a leftover leaf and compile as the constant $3$ or the
    identity.
27. `affine_scale` fits Keijzer $a + b\,f(x)$ on the same
    `valid=` mask as `case_errors`. Darwinian callers use the
    scaled series only for fitness / case errors. Lamarckian
    `write_affine_scale` writes $a$ and $b$ back as ephemerals
    wrapping a `PrimitiveTree`, or as wrapping Slim deltas, then
    invalidates fitness and the compile cache for that expression.
28. <a id="28-private-push-gp-policy-loop"></a>`LinearPolicyProgram`,
    `PushPolicyProgram`, and `step_policy_loop` implement the
    private Push GP loop behind the P11–P14 firewall.
    `policy_observe` supplies summary observations;
    `linear_policy_decide` / `push_policy_decide` emit discrete
    action tokens; `apply_policy_action` dispatches them. No
    column loads, no `Window` Push type, and no public
    `PushTree` on `gp` or `tools`. Tapes remain the only
    `interpret_tapes` target. Not a second public genome
    ([Push GP P19](../roadmap/push_gp.md#p19-push-gp-as-the-loop)).
29. `interpret_tapes` hash-conses postfix subexpressions across a
    batch and evaluates each unique sub-tape once against the
    packed matrix. Shared suffixes are stitched from one oracle
    result per node. The return shape, warmup ``nan`` contract,
    and per-tape ``fill`` semantics are unchanged. See the
    [columnar GP tutorial](../../tutorials/columnar_gp.md).
30. HARM `natural_histogram` does not wrap `hist[-1]` when a
    tree has size $0$. The left-neighbor bin is updated only
    for `ind_size >= 1`, matching the existing `ind_size - 2`
    guard.
31. `register_gp` wires the standard tree-GP toolbox:
    `clone_individual`, `compile_tree`, half-and-half init,
    one-point crossover, uniform mutation, and a height
    `static_limit`. `columnar_pset` builds the typed column set
    and registers the NumPy / window kits (optional pair-window,
    time-series, and window ephemeral) in one call.
    `evaluate_columnar` is the `evaluate_batch` helper: unique
    trees are lowered once, scored with `interpret_tapes`, and
    warmup `nan` samples are dropped from the MSE. DEAP leaves
    that wiring on every caller.

The columnar contract is in the
[columnar GP tutorial](../../tutorials/columnar_gp.md). The private
Push policy loop is in the
[Push GP tutorial](../../tutorials/push_gp.md). Shared-array
evaluation is in the
[multiprocessing tutorial](../../tutorials/multiprocessing.md).

[deap-24]: https://github.com/DEAP/deap/issues/24
[deap-383]: https://github.com/DEAP/deap/issues/383
[deap-644]: https://github.com/DEAP/deap/issues/644
