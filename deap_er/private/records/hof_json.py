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
import json
from typing import TYPE_CHECKING, Any

from .logbook import _json_ready

if TYPE_CHECKING:
    from .hall_of_fame import HallOfFame

__all__: list[str] = ["hall_of_fame_from_json", "hall_of_fame_to_json"]


def hall_of_fame_to_json(hof: "HallOfFame") -> str:
    """Serialize a hall of fame to JSON.

    Each member stores ``genes`` and ``fitness``. NumPy scalars become
    Python numbers. The ``similar`` predicate is not serialized.

    Args:
        hof: Archive to serialize.

    Returns:
        A JSON document with ``maxsize`` and member records.
    """
    payload = {
        "maxsize": hof.maxsize,
        "items": [
            {
                "genes": _json_ready(list(individual)),
                "fitness": _json_ready(list(individual.fitness.values)),
            }
            for individual in hof.items
        ],
    }
    return json.dumps(payload)


def hall_of_fame_from_json(
    text: str, ind_cls: type[Any] | None, hof_cls: type["HallOfFame"]
) -> "HallOfFame":
    """Rebuild a hall of fame from JSON.

    Args:
        text: JSON document from :func:`hall_of_fame_to_json`.
        ind_cls: Creator individual type used to rebuild members.
            Required when the archive is non-empty.
        hof_cls: ``HallOfFame`` class used to construct the archive.

    Returns:
        A hall of fame with restored members in best-first order.

    Raises:
        ValueError: If members are present and ``ind_cls`` is omitted.
    """
    data = json.loads(text)
    hof = hof_cls(maxsize=int(data["maxsize"]))
    items = data.get("items", [])
    if items:
        if ind_cls is None:
            raise ValueError("ind_cls is required to restore a non-empty hall of fame")
        for item in items:
            individual = ind_cls(list(item["genes"]))
            individual.fitness.values = tuple(item["fitness"])
            hof.insert(individual)
    return hof
