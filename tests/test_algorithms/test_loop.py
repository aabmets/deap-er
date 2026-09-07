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
import logging
from typing import Any, override

from deap_er import Fitness, Toolbox, creator, tools

LOOP_FIT = "LOOP_FIT"
LOOP_IND = "LOOP_IND"


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_ea_simple_log_time_logger_and_fronts():
    creator.create_type(LOOP_FIT, Fitness, weights=(-1.0,))
    creator.create_type(LOOP_IND, list, fitness=creator.__dict__[LOOP_FIT])
    try:
        ind_cls = creator.__dict__[LOOP_IND]
        toolbox = Toolbox()
        toolbox.register("evaluate", lambda individual: (sum(individual),))
        toolbox.register("select", tools.sel_random)
        toolbox.register("mate", tools.cx_one_point)
        toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.0)

        logger = logging.getLogger("deap_er.test.loop")
        handler = _ListHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        fronts: list[Any] = []
        population = [ind_cls([0, 1, 0, 1]) for _ in range(6)]
        _, timed = tools.ea_simple(
            toolbox,
            population,
            generations=2,
            cx_prob=0.5,
            mut_prob=0.0,
            verbose=True,
            logger=logger,
            log_time=True,
            fronts=fronts,
        )
        assert timed.header[:3] == ["gen", "nevals", "duration"]
        assert all("duration" in entry for entry in timed)
        assert len(fronts) == 3
        assert handler.messages

        other = [ind_cls([1, 0, 1, 0]) for _ in range(6)]
        _, plain = tools.ea_simple(toolbox, other, generations=1, cx_prob=0.0, mut_prob=0.0)
        assert plain.header == ["gen", "nevals"]
        assert all("duration" not in entry for entry in plain)
    finally:
        del creator.__dict__[LOOP_FIT]
        del creator.__dict__[LOOP_IND]


def test_ea_simple_empty_population_with_max_stats():
    creator.create_type(LOOP_FIT, Fitness, weights=(-1.0,))
    creator.create_type(LOOP_IND, list, fitness=creator.__dict__[LOOP_FIT])
    try:
        toolbox = Toolbox()
        toolbox.register("evaluate", lambda individual: (sum(individual),))
        toolbox.register("select", tools.sel_random)
        toolbox.register("mate", tools.cx_one_point)
        toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.0)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("max", max)

        _, logbook = tools.ea_simple(
            toolbox, [], generations=0, cx_prob=0.0, mut_prob=0.0, stats=stats
        )
        assert logbook.select("gen") == [0]
    finally:
        del creator.__dict__[LOOP_FIT]
        del creator.__dict__[LOOP_IND]
