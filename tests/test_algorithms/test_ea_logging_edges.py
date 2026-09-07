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
from typing import override

from deap_er import Fitness, Toolbox, creator, tools

ME_FIT = "LOG_ME_FIT"
ME_IND = "LOG_ME_IND"
RST_FIT = "LOG_RST_FIT"
RST_IND = "LOG_RST_IND"


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _evaluate(individual):
    return (sum(individual),)


def _behavior(individual):
    return (float(len(individual)),)


def test_ea_map_elites_verbose_logger_and_duration():
    creator.create_type(ME_FIT, Fitness, weights=(1.0,))
    creator.create_type(ME_IND, list, fitness=creator.__dict__[ME_FIT])
    try:
        toolbox = Toolbox()
        toolbox.register("mate", tools.cx_two_point)
        toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.2)
        toolbox.register("evaluate", _evaluate)
        tools.rng.seed(7)
        initial = [creator.__dict__[ME_IND]([0, 1, 1, 0, 1, 0]) for _ in range(6)]
        archive = tools.GridArchive(ranges=[(6.0, 7.0)], bins=2)
        logger = logging.getLogger("deap_er.test.map_elites")
        handler = _ListHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        _, logbook = tools.ea_map_elites(
            toolbox,
            archive,
            _behavior,
            initial,
            generations=1,
            batch_size=4,
            cx_prob=0.0,
            mut_prob=0.2,
            verbose=True,
            logger=logger,
            log_time=True,
        )

        assert handler.messages
        assert "duration" in logbook[0]
    finally:
        del creator.__dict__[ME_FIT]
        del creator.__dict__[ME_IND]


def test_ea_map_elites_verbose_prints(capsys):
    creator.create_type(ME_FIT, Fitness, weights=(1.0,))
    creator.create_type(ME_IND, list, fitness=creator.__dict__[ME_FIT])
    try:
        toolbox = Toolbox()
        toolbox.register("mate", tools.cx_two_point)
        toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.2)
        toolbox.register("evaluate", _evaluate)
        initial = [creator.__dict__[ME_IND]([1, 0, 1, 0, 1, 1]) for _ in range(4)]
        archive = tools.GridArchive(ranges=[(6.0, 7.0)], bins=2)

        tools.ea_map_elites(
            toolbox,
            archive,
            _behavior,
            initial,
            generations=0,
            batch_size=4,
            cx_prob=0.0,
            mut_prob=0.0,
            verbose=True,
        )

        assert capsys.readouterr().out
    finally:
        del creator.__dict__[ME_FIT]
        del creator.__dict__[ME_IND]


def test_ea_generate_update_restarts_verbose_and_empty_generate(capsys):
    creator.create_type(RST_FIT, Fitness, weights=(-1.0,))
    creator.create_type(RST_IND, list, fitness=creator.__dict__[RST_FIT])
    try:
        strategy = tools.Strategy(centroid=[0.0] * 3, sigma=1.0, offsprings=4)
        restart = tools.RestartStrategy(strategy, mode="ipop", budget=8)
        toolbox = Toolbox()
        toolbox.register("evaluate", tools.bm_sphere)
        toolbox.register("generate", restart.generate, creator.__dict__[RST_IND])
        toolbox.register("update", restart.update)
        logger = logging.getLogger("deap_er.test.restarts")
        handler = _ListHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        tools.ea_generate_update_restarts(
            toolbox, restart, verbose=True, logger=logger, log_restarts=False
        )
        assert handler.messages

        restart_print = tools.RestartStrategy(
            tools.Strategy(centroid=[0.0] * 3, sigma=1.0, offsprings=4),
            mode="ipop",
            budget=8,
        )
        toolbox.register("generate", restart_print.generate, creator.__dict__[RST_IND])
        toolbox.register("update", restart_print.update)
        tools.ea_generate_update_restarts(toolbox, restart_print, verbose=True, log_restarts=False)
        assert capsys.readouterr().out

        spent = tools.RestartStrategy(
            tools.Strategy(centroid=[0.0] * 3, sigma=1.0, offsprings=4),
            mode="ipop",
            budget=0,
        )
        toolbox.register("generate", spent.generate, creator.__dict__[RST_IND])
        toolbox.register("update", spent.update)
        population, _ = tools.ea_generate_update_restarts(toolbox, spent, log_restarts=False)
        assert population == []
    finally:
        del creator.__dict__[RST_FIT]
        del creator.__dict__[RST_IND]
