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
from typing import Any

import pytest
from deap_er import tools


class _Ind(list[Any]):
    pass


def test_history_update_and_genealogy_walk():
    history = tools.History()
    parents: Any = [_Ind([0]), _Ind([1])]
    history.update(parents)

    child: Any = _Ind([2])
    child.history_index = parents[0].history_index

    def mate(_first: Any, _second: Any) -> tuple[Any]:
        return (child,)

    wrapped = history.decorator(mate)
    wrapped(parents[0], parents[1])

    tree = history.get_genealogy(child)
    assert child.history_index in tree
    assert history.genealogy_index == 3

    shallow = history.get_genealogy(child, max_depth=0)
    assert shallow == {}


def test_get_genealogy_requires_history_index():
    history = tools.History()
    missing: Any = _Ind([0])
    with pytest.raises(AttributeError, match="history_index"):
        history.get_genealogy(missing)
