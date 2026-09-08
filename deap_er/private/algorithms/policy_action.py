#
#   Apache License 2.0
#
#   Copyright (c) 2022, Mattias Aabmets
#
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#
#   SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from deap_er.private.algorithms.loop import evaluate_invalid
from deap_er.private.algorithms.step_islands import step_islands
from deap_er.private.operators.case_exam_step import next_lexicase_cases
from deap_er.private.operators.policy_action_guard import (
    PolicyActionGuard,
    estimate_policy_action_evals,
    guard_policy_action,
)
from deap_er.private.programming.memetic import tune_ephemerals
from deap_er.private.programming.promote import promote_subtree
from deap_er.private.programming.tape_batch import interpret_tapes

POLICY_ACTION_SKIP_TUNE = "skip_tune"
POLICY_ACTION_SKIP_PROMOTE = "skip_promote"

SKIP_POLICY_ACTIONS: frozenset[str] = frozenset(
    {
        POLICY_ACTION_SKIP_TUNE,
        POLICY_ACTION_SKIP_PROMOTE,
    }
)

SUPPORTED_POLICY_ACTIONS: frozenset[str] = frozenset(
    {
        "next_lexicase_cases",
        "tune_ephemerals",
        POLICY_ACTION_SKIP_TUNE,
        "promote_subtree",
        POLICY_ACTION_SKIP_PROMOTE,
        "evaluate_invalid",
        "interpret_tapes",
        "step_islands",
    }
)

__all__: list[str] = [
    "POLICY_ACTION_SKIP_PROMOTE",
    "POLICY_ACTION_SKIP_TUNE",
    "SKIP_POLICY_ACTIONS",
    "SUPPORTED_POLICY_ACTIONS",
    "PolicyActionGuard",
    "PolicyActionResult",
    "apply_policy_action",
    "estimate_policy_action_evals",
    "guard_policy_action",
]


@dataclass(frozen=True, slots=True)
class PolicyActionResult:
    """Outcome of :func:`apply_policy_action`.

    Attributes:
        applied: True when an underlying callable ran.
        rejected: True when the token is unknown, required kwargs were
            missing, or a ``guard`` cap rejected the action.
        value: Return value from the underlying callable when
            ``applied`` is True; otherwise ``None``.
    """

    applied: bool
    rejected: bool
    value: Any = None


def apply_policy_action(action: str, /, **kwargs: Any) -> PolicyActionResult:
    """Map a discrete policy action token onto existing toolbox callables.

    Push emits action names, not trees. This helper is schema plus
    thin dispatch only: fitness assignment and rescore ownership stay
    on the caller. Skip tokens are intentional no-ops. Unknown tokens,
    missing required kwargs, and guard cap violations are rejected
    without raising.

    Args:
        action: One of :data:`SUPPORTED_POLICY_ACTIONS`.
        **kwargs: Arguments forwarded to the underlying callable for
            the chosen action. See that function's docstring. Optional
            ``guard`` (:class:`~deap_er.operators.PolicyActionGuard`)
            enforces action caps from
            :func:`~deap_er.operators.guard_policy_action`.

    Returns:
        A :class:`PolicyActionResult` describing whether the action
        ran, was skipped, or was rejected.
    """
    guard = kwargs.pop("guard", None)
    if action in SKIP_POLICY_ACTIONS:
        return PolicyActionResult(applied=False, rejected=False)
    if guard is not None and not guard_policy_action(action, guard, **kwargs):
        return PolicyActionResult(applied=False, rejected=True)
    if action == "next_lexicase_cases":
        result = _dispatch_next_lexicase_cases(**kwargs)
    elif action == "tune_ephemerals":
        result = _dispatch_tune_ephemerals(**kwargs)
    elif action == "promote_subtree":
        result = _dispatch_promote_subtree(**kwargs)
    elif action == "evaluate_invalid":
        result = _dispatch_evaluate_invalid(**kwargs)
    elif action == "interpret_tapes":
        result = _dispatch_interpret_tapes(**kwargs)
    elif action == "step_islands":
        result = _dispatch_step_islands(**kwargs)
    else:
        return PolicyActionResult(applied=False, rejected=True)
    if guard is not None and result.applied:
        evals = _applied_eval_cost(action, result, **kwargs)
        guard.note_applied(action, evals=evals)
    return result


def _applied_eval_cost(action: str, result: PolicyActionResult, **kwargs: Any) -> int:
    if action == "evaluate_invalid" and isinstance(result.value, int):
        return result.value
    return estimate_policy_action_evals(action, **kwargs)


def _reject() -> PolicyActionResult:
    return PolicyActionResult(applied=False, rejected=True)


def _apply(value: Any) -> PolicyActionResult:
    return PolicyActionResult(applied=True, rejected=False, value=value)


def _missing(required: Sequence[str], kwargs: dict[str, Any]) -> bool:
    return any(name not in kwargs for name in required)


def _dispatch_next_lexicase_cases(**kwargs: Any) -> PolicyActionResult:
    if _missing(("exams", "elites"), kwargs):
        return _reject()
    value = next_lexicase_cases(
        kwargs["exams"],
        kwargs["elites"],
        matrix=kwargs.get("matrix"),
        trust_matrix=kwargs.get("trust_matrix", False),
        solved=kwargs.get("solved"),
        case_count=kwargs.get("case_count"),
        informed=kwargs.get("informed", True),
        mut_prob=kwargs.get("mut_prob", 0.2),
        mode=kwargs.get("mode", "unsolved"),
        held_out=kwargs.get("held_out"),
        min_cases=kwargs.get("min_cases", 1),
        length=kwargs.get("length"),
    )
    return _apply(value)


def _dispatch_tune_ephemerals(**kwargs: Any) -> PolicyActionResult:
    if _missing(("individual", "strategy"), kwargs):
        return _reject()
    if kwargs.get("evaluate") is None and kwargs.get("evaluate_batch") is None:
        return _reject()
    value = tune_ephemerals(
        kwargs["individual"],
        kwargs["strategy"],
        evaluate=kwargs.get("evaluate"),
        n_gen=kwargs.get("n_gen", 5),
        evaluate_batch=kwargs.get("evaluate_batch"),
        clone=kwargs.get("clone"),
    )
    return _apply(value)


def _dispatch_promote_subtree(**kwargs: Any) -> PolicyActionResult:
    if _missing(("prim_set", "expr"), kwargs):
        return _reject()
    value = promote_subtree(
        kwargs["prim_set"],
        kwargs["expr"],
        index=kwargs.get("index", 0),
        max_library=kwargs.get("max_library", 32),
        prefix=kwargs.get("prefix", "promo"),
        weight=kwargs.get("weight", 1.0),
    )
    return _apply(value)


def _dispatch_evaluate_invalid(**kwargs: Any) -> PolicyActionResult:
    if _missing(("toolbox", "individuals"), kwargs):
        return _reject()
    value = evaluate_invalid(kwargs["toolbox"], kwargs["individuals"])
    return _apply(value)


def _dispatch_interpret_tapes(**kwargs: Any) -> PolicyActionResult:
    if _missing(("tapes", "matrix"), kwargs):
        return _reject()
    value = interpret_tapes(
        kwargs["tapes"],
        kwargs["matrix"],
        backend=kwargs.get("backend", "opcode"),
        dispatch=kwargs.get("dispatch"),
        parallel=kwargs.get("parallel", False),
    )
    return _apply(value)


def _dispatch_step_islands(**kwargs: Any) -> PolicyActionResult:
    if _missing(("demes",), kwargs):
        return _reject()
    step_islands(
        kwargs["demes"],
        kwargs.get("migrate"),
        eval_keys=kwargs.get("eval_keys"),
    )
    return _apply(None)
