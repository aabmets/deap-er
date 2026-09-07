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
import math
from typing import Any, cast

import numpy
import pytest
from deap_er import Toolbox, tools
from deap_er.private.operators.constraint_dominates import sort_constraint_dominated


def _feas(index):
    return lambda ind: ind[0] == index


def _cv(mapping):
    return lambda ind: mapping[ind[0]]


def test_feasible_beats_infeasible(multi_obj, make):
    good = make(multi_obj, [0], (1.0, 1.0))
    bad = make(multi_obj, [1], (10.0, 10.0))
    assert tools.constraint_dominates(good, bad, feasible=_feas(0))
    assert not tools.constraint_dominates(bad, good, feasible=_feas(0))


def test_two_feasibles_use_pareto(multi_obj, make):
    better = make(multi_obj, [0], (5.0, 5.0))
    worse = make(multi_obj, [1], (1.0, 1.0))
    other = make(multi_obj, [2], (5.0, 0.0))
    assert tools.constraint_dominates(better, worse, feasible=lambda _ind: True)
    assert not tools.constraint_dominates(worse, better, feasible=lambda _ind: True)
    assert not tools.constraint_dominates(other, better, feasible=lambda _ind: True)
    assert tools.constraint_dominates(better, other, feasible=lambda _ind: True)
    left = make(multi_obj, [3], (5.0, 1.0))
    right = make(multi_obj, [4], (1.0, 5.0))
    assert not tools.constraint_dominates(left, right, feasible=lambda _ind: True)
    assert not tools.constraint_dominates(right, left, feasible=lambda _ind: True)


def test_two_infeasibles_prefer_smaller_violation(multi_obj, make):
    low = make(multi_obj, [0], (1.0, 1.0))
    high = make(multi_obj, [1], (9.0, 9.0))
    viol = _cv({0: 0.5, 1: 2.0})
    assert tools.constraint_dominates(low, high, violation=viol)
    assert not tools.constraint_dominates(high, low, violation=viol)


def test_equal_violation_neither_dominates(multi_obj, make):
    left = make(multi_obj, [0], (9.0, 1.0))
    right = make(multi_obj, [1], (1.0, 9.0))
    assert not tools.constraint_dominates(left, right, violation=lambda _ind: 1.5)
    assert not tools.constraint_dominates(right, left, violation=lambda _ind: 1.5)


def test_violation_only_nonpositive_is_feasible(multi_obj, make):
    slack = make(multi_obj, [0], (1.0, 1.0))
    bound = make(multi_obj, [1], (2.0, 2.0))
    breach = make(multi_obj, [2], (9.0, 9.0))
    viol = _cv({0: -1.0, 1: 0.0, 2: 0.1})
    assert tools.constraint_dominates(slack, breach, violation=viol)
    assert tools.constraint_dominates(bound, breach, violation=viol)
    assert not tools.constraint_dominates(slack, bound, violation=viol)


def test_feasible_only_infeasibles_do_not_dominate(multi_obj, make):
    low = make(multi_obj, [0], (9.0, 9.0))
    high = make(multi_obj, [1], (1.0, 1.0))
    assert not tools.constraint_dominates(low, high, feasible=lambda _ind: False)
    assert not tools.constraint_dominates(high, low, feasible=lambda _ind: False)


def test_flag_wins_when_both_given(multi_obj, make):
    flagged = make(multi_obj, [0], (1.0, 1.0))
    clean = make(multi_obj, [1], (2.0, 2.0))
    kwargs = {"feasible": _feas(1), "violation": _cv({0: 0.0, 1: 4.0})}
    assert tools.constraint_dominates(clean, flagged, **kwargs)
    assert not tools.constraint_dominates(flagged, clean, **kwargs)


def test_requires_a_constraint_callable(multi_obj, make):
    ind = make(multi_obj, [0], (1.0, 1.0))
    with pytest.raises(TypeError, match="feasible or violation"):
        tools.constraint_dominates(ind, ind)


def test_rejects_non_callable_kwargs(multi_obj, make):
    ind = make(multi_obj, [0], (1.0, 1.0))
    not_feasible = cast(Any, True)
    not_violation = cast(Any, 1.0)
    with pytest.raises(TypeError, match="feasible"):
        tools.constraint_dominates(ind, ind, feasible=not_feasible)
    with pytest.raises(TypeError, match="violation"):
        tools.constraint_dominates(ind, ind, violation=not_violation)


def test_rejects_non_scalar_and_nonfinite_violation(multi_obj, make):
    ind = make(multi_obj, [0], (1.0, 1.0))
    vector_violation = cast(Any, lambda _ind: (1.0, 2.0))
    with pytest.raises(TypeError, match="real scalar"):
        tools.constraint_dominates(ind, ind, violation=vector_violation)
    with pytest.raises(ValueError, match="finite"):
        tools.constraint_dominates(ind, ind, violation=lambda _ind: math.nan)


def test_sel_nsga_2_empty_and_nonpositive_count(multi_obj, make):
    population = [make(multi_obj, [0], (1.0, 1.0))]
    assert tools.sel_nsga_2([], 3, feasible=lambda _ind: True) == []
    assert tools.sel_nsga_2(population, 0, violation=lambda _ind: 0.0) == []
    assert tools.sel_nsga_2(population, -1, feasible=lambda _ind: True) == []


def test_all_feasible_matches_unconstrained(multi_obj, make):
    population = [
        make(multi_obj, [0], (9.0, 1.0)),
        make(multi_obj, [1], (8.0, 2.0)),
        make(multi_obj, [2], (1.0, 9.0)),
        make(multi_obj, [3], (2.0, 8.0)),
        make(multi_obj, [4], (5.0, 5.0)),
    ]
    plain = tools.sel_nsga_2(population, 3)
    constrained = tools.sel_nsga_2(population, 3, feasible=lambda _ind: True)
    assert [ind[0] for ind in constrained] == [ind[0] for ind in plain]


def test_feasibles_selected_before_infeasibles(multi_obj, make):
    population = [
        make(multi_obj, [0], (2.0, 1.0)),
        make(multi_obj, [1], (1.0, 2.0)),
        make(multi_obj, [2], (9.0, 9.0)),
    ]
    chosen = tools.sel_nsga_2(population, 2, feasible=lambda ind: ind[0] < 2)
    assert sorted(ind[0] for ind in chosen) == [0, 1]


def test_infeasible_tail_orders_by_violation(multi_obj, make):
    population = [
        make(multi_obj, [0], (1.0, 1.0)),
        make(multi_obj, [1], (8.0, 8.0)),
        make(multi_obj, [2], (9.0, 9.0)),
        make(multi_obj, [3], (7.0, 7.0)),
    ]
    chosen = tools.sel_nsga_2(
        population, 3, feasible=_feas(0), violation=_cv({0: 0.0, 1: 3.0, 2: 1.0, 3: 2.0})
    )
    assert [ind[0] for ind in chosen] == [0, 2, 3]


def test_fitness_values_are_not_rewritten(multi_obj, make):
    population = [
        make(multi_obj, [0], (1.0, 4.0)),
        make(multi_obj, [1], (2.0, 3.0)),
        make(multi_obj, [2], (3.0, 2.0)),
    ]
    before = [ind.fitness.values for ind in population]
    tools.sel_nsga_2(population, 2, violation=_cv({0: 1.0, 1: 0.0, 2: 2.0}))
    assert [ind.fitness.values for ind in population] == before


def test_toolbox_register_passes_constraint_kwargs(multi_obj, make):
    population = [
        make(multi_obj, [0], (1.0, 1.0)),
        make(multi_obj, [1], (8.0, 8.0)),
        make(multi_obj, [2], (9.0, 9.0)),
    ]
    toolbox = Toolbox()
    toolbox.register(
        "select", tools.sel_nsga_2, feasible=_feas(0), violation=_cv({0: 0.0, 1: 2.0, 2: 1.0})
    )
    chosen = toolbox.select(population, 2)
    assert [ind[0] for ind in chosen] == [0, 2]


def test_equal_cv_infeasibles_share_a_front(multi_obj, make):
    population = [
        make(multi_obj, [0], (9.0, 1.0)),
        make(multi_obj, [1], (1.0, 9.0)),
        make(multi_obj, [2], (2.0, 2.0)),
    ]
    fronts = sort_constraint_dominated(population, 3, violation=_cv({0: 1.0, 1: 1.0, 2: 4.0}))
    assert [ind[0] for ind in fronts[0]] == [0, 1]
    assert [ind[0] for ind in fronts[1]] == [2]


def test_accepts_numpy_scalar_violation(multi_obj, make):
    low = make(multi_obj, [0], (1.0, 1.0))
    high = make(multi_obj, [1], (2.0, 2.0))
    viol = _cv({0: numpy.float64(0.25), 1: numpy.float64(1.5)})
    assert tools.constraint_dominates(low, high, violation=viol)
