# Push GP

Push GP in deap-er is a **two-level** search. Columnar tapes stay the
laws: they are the only programs `interpret_tapes` runs. A policy
reads summaries and emits discrete toolbox actions — which exam to
score next, whether to tune or promote, when to rescore. That is not
a second public genome. Prefix trees plus the tape remain the
program. A public `PushTree` on `gp` / `tools` is
[not planned](../overview/roadmap/not_planned.md).

This tutorial covers the published firewall — observe, act, score
the policy on held-out cases, and cap spend — and where the private
interpreter sits. The laws themselves are in the
[columnar programs](columnar_gp.md) tutorial. The backlog that
assembled this path is the [Push GP](../overview/roadmap/push_gp.md)
roadmap page.

!!! attention
    `PushPolicyProgram`, `LinearPolicyProgram`, and
    `step_policy_loop` live under `deap_er.private.programming`.
    They are not in `gp.__all__` or `tools`. The public surface is
    `policy_observe` in, `apply_policy_action` out.

## The contract

The policy never sees a column, a `Window`, or `matrix[t]`.
`policy_observe` rejects a raw NumPy pack at the boundary. The
only typed layout it may read is
[`PolicyObservation`](../reference/records.md):

| Field | Meaning |
|:------|:--------|
| `solve_bits` | per-case `0/1` flags for the observed tape |
| `unsolved_count` | zeros in `solve_bits` |
| `train_score` | sum of train-exam difficulties |
| `held_out_score` | held-out difficulty, or `None` |
| `archive_coverage` / `qd_score` | from `ArchiveStats` |
| `nevals` / `rows_seen` | budget and matrix progress |
| `promoted_library_size` | names from `promoted_names` |
| `fitness_invalid` | the observed tape has no valid fitness |
| `last_action_rejected` | the previous action hit a guard |

Build that record from summaries you already have — case errors,
exam scores, archive stats — not from the evaluation matrix.

```python
from deap_er import gp, records, tools

errors = tools.case_errors(predicted, target, ranges)
solve_bits = tools.policy_solve_bits_from_errors(errors)
train_score, held_out_score = tools.policy_exam_scores(pool, elites)

obs = tools.policy_observe(
    solve_bits=solve_bits,
    train_score=train_score,
    held_out_score=held_out_score,
    archive=records.ArchiveStats(
        num_elites=2,
        num_cells=4,
        coverage=0.5,
        qd_score=3.0,
    ),
    nevals=7,
    rows_seen=128,
    promoted_library_size=tools.policy_promoted_library_size(pset),
)
```

`policy_solve_bits_from_fitness` uses the same zero threshold as
lexicase. `policy_solve_bits_from_semantic_row` accepts one row of
`gp.semantic_solve_bits`. `obs.as_tuple()` is the fixed field order
a linear or Push policy may consume.

## Actions

`apply_policy_action` maps one token onto an existing callable.
Push emits names, not trees. Unknown tokens and missing kwargs are
rejected without raising.

| Token | Dispatches to |
|:------|:--------------|
| `next_lexicase_cases` | `next_lexicase_cases` (set the next `cases=` exam) |
| `tune_ephemerals` | `gp.tune_ephemerals` |
| `skip_tune` | no-op |
| `promote_subtree` | `gp.promote_subtree` |
| `skip_promote` | no-op |
| `evaluate_invalid` | evaluate individuals whose fitness is invalid |
| `interpret_tapes` | `gp.interpret_tapes` |
| `step_islands` | `tools.step_islands` |

```python
result = tools.apply_policy_action(
    "next_lexicase_cases",
    exams=pool.exams,
    elites=elites,
    mut_prob=0.0,
)
if result.rejected:
    # last_action_rejected=True on the next observe
    ...
elif result.applied:
    next_exam = result.value
```

Skip tokens return `applied=False` and `rejected=False`. Fitness
assignment stays on the caller — this helper is schema plus
dispatch. `ea_policy` is the thin `ea_simple` wrapper that calls
`begin_generation`, observe → decide → apply, then select / vary /
evaluate. It is not `step_program_search`: no Slim, tune, archive,
or team composition is baked in. See
[Operators and Algorithms](operators_and_algorithms.md) and the
[algorithms](../reference/algorithms.md) reference.

## Held-out policy fitness

Score the *policy* only on a caller-marked held-out exam.
Train-exam quality is an observation. Otherwise the policy evolves
“make the exam easy.”

```python
tools.guard_policy_fitness_exam(
    fitness_exam,
    held_out=pool.held_out,
    n_cases=4,
    train_exams=pool.exams,
)
quality = tools.policy_held_out_fitness(elites, pool)
tools.record_policy_generalization_gap(
    logbook,
    gen=gen,
    train_score=train_score,
    held_out_score=held_out_score,
)
```

`guard_policy_fitness_exam` refuses a train exam or an exam the
policy just mutated. Chronological meaning of “held-out” stays on
you. The logbook chapter is `generalization_gap` (`train`,
`held_out`, `gap`).

## Guards and budget

`PolicyActionGuard` caps promote rate, inner `tune` generations,
minimum exam size, promote cooldown, and remaining `n_evals`. A
rejected action is an observation, not a crash. Call
`begin_generation` at the start of each outer generation so
per-generation promote limits reset.

```python
guard = tools.PolicyActionGuard(
    max_promotes_per_gen=1,
    max_tune_gen=5,
    n_evals=10_000,
)
guard.begin_generation()
result = tools.apply_policy_action(
    "promote_subtree",
    prim_set=pset,
    expr=tree,
    guard=guard,
)
```

`estimate_policy_action_evals` is the conservative cost used when
the guard still has budget. Pair that with `EvalCache` and the
`n_evals=` stop on `ea_simple` / `ea_map_elites` so a policy that
tunes every generation cannot win by spending. Worker streams for
two populations (tapes and policies) are in
[Multiprocessing](multiprocessing.md).

## A public decide callable

The firewall does not require the private interpreter. Any
`PolicyObservation -> str` function is a valid policy — including
a hand-written rule or a linear decision list.

```python
def decide(obs: records.PolicyObservation) -> str:
    if obs.last_action_rejected:
        return tools.POLICY_ACTION_SKIP_TUNE
    if obs.unsolved_count > 1:
        return "next_lexicase_cases"
    return tools.POLICY_ACTION_SKIP_PROMOTE

obs = tools.policy_observe(
    solve_bits=(1, 0, 0, 1),
    train_score=2.0,
    held_out_score=1.0,
)
action = decide(obs)
result = tools.apply_policy_action(
    action,
    exams=pool.exams,
    elites=elites,
    mut_prob=0.0,
    guard=guard,
)
```

If a uniform random policy over `SUPPORTED_POLICY_ACTIONS` cannot
run for thousands of generations without melting the compile
cache, do not add a Push stack.

## The private interpreter

The shipped Push individual is a tiny instruction set on `int` /
`bool` / a short solve-bit vector. It reads `policy_observe` and
emits one of the action tokens above. No column loads, no `Window`
as a Push type, no per-row Push, no `rolling_mean` as a Push
instruction.

```python
from deap_er.private.programming.policy_loop import step_policy_loop
from deap_er.private.programming.policy_push import (
    EMIT,
    LOAD_UNSOLVED,
    LT,
    PUSH_BOOL,
    PUSH_INT,
    PushPolicyProgram,
    push_policy_decide,
)

program = PushPolicyProgram(
    code=(LOAD_UNSOLVED, PUSH_INT, 2, LT, PUSH_BOOL, 1, EMIT),
)
step = step_policy_loop(
    lambda obs: push_policy_decide(program, obs),
    solve_bits=(1, 0, 0, 1),
    train_score=2.0,
    held_out_score=1.0,
    action_kwargs={"exams": pool.exams, "elites": elites, "mut_prob": 0.0},
    guard=guard,
)
```

`LinearPolicyProgram` is the same interface without an Exec stack
— the acceptance test that the firewall works. Both types stay
private on purpose: tapes remain the only `interpret_tapes`
target.

## What stays on you

Evaluation, data loading, and any domain metric stay on the
caller, same as every other toolbox loop. The library owns the
observation schema, the action tokens, the held-out scoring
convention, the caps, and the thin `ea_policy` driver. It does
not own a `step_program_search` that writes fitness.

Related reference: [utilities](../reference/utilities.md),
[operators](../reference/operators.md),
[algorithms](../reference/algorithms.md),
[genetic programming](../reference/gp.md).
A complete script is the
[Push GP example](../examples/genetic_programming/push_gp.md).
