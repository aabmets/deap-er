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

from typing import Literal

from deap_er.private.strategies.restart_common import sample_small_lambda, sample_small_sigma

__all__: list[str] = [
    "RestartSchedule",
    "next_bipop_params",
    "next_ipop_params",
    "validate_lambda_factor",
]

type Regime = Literal["large", "small"]


class RestartSchedule:
    """IPOP / BIPOP population size, step size, and regime budgets.

    Holds only scheduling state. ``RestartStrategy`` owns the wrapped
    strategy, the evaluation counters, and termination.

    Args:
        mode: ``ipop`` or ``bipop``.
        lambda_default: First-run offspring count.
        lambda_factor: Population growth factor per large restart.
        max_large_restarts: Cap on the large-restart exponent.
        sigma_large: First-run / large-regime step size.

    Raises:
        ValueError: If ``lambda_factor`` can make a restart λ below 1.
    """

    def __init__(
        self,
        mode: Literal["ipop", "bipop"],
        lambda_default: int,
        lambda_factor: float,
        max_large_restarts: int,
        sigma_large: float,
    ) -> None:
        """See the class docstring."""
        validate_lambda_factor(lambda_default, lambda_factor, max_large_restarts)
        self.mode = mode
        self.lambda_default = lambda_default
        self.lambda_factor = lambda_factor
        self.max_large_restarts = max_large_restarts
        self.sigma_large = sigma_large
        self.lambda_large = lambda_default
        self.irestart_large = 0
        self.budget_large = 0
        self.budget_small = 0
        self.last_large_run_evals = 0
        self.regime: Regime | None = None

    def account_run(self, run_evals: int) -> None:
        """Charge a finished run's evaluations to its regime.

        Args:
            run_evals: Evaluations the finished run used. The first
                run counts as large.
        """
        if run_evals == 0:
            return
        if self.regime == "small":
            self.budget_small += run_evals
        else:
            self.budget_large += run_evals
            self.last_large_run_evals = run_evals

    def next_run(
        self, restart_count: int, evals_used: int, budget: int
    ) -> tuple[int, float, int | None]:
        """Pick the next regime and return its λ, σ, and evaluation cap.

        Args:
            restart_count: Restarts including the one being started.
            evals_used: Evaluations used so far across all runs.
            budget: Total evaluation budget.

        Returns:
            Offspring count, step size, and the small-run evaluation cap
            (half the last large run), or None for a large run.
        """
        if self.mode == "ipop":
            lamb, sigma, self.irestart_large = next_ipop_params(
                self.lambda_default,
                self.lambda_factor,
                self.irestart_large,
                self.max_large_restarts,
                self.sigma_large,
            )
            self.regime = "large"
        else:
            lamb, sigma, self.regime, self.irestart_large, self.lambda_large = next_bipop_params(
                lambda_default=self.lambda_default,
                lambda_factor=self.lambda_factor,
                lambda_large=self.lambda_large,
                irestart_large=self.irestart_large,
                max_large_restarts=self.max_large_restarts,
                sigma_large=self.sigma_large,
                restart_count=restart_count,
                evals_used=evals_used,
                budget=budget,
                budget_large=self.budget_large,
                budget_small=self.budget_small,
            )
        if self.regime == "small":
            return lamb, sigma, max(1, self.last_large_run_evals // 2)
        return lamb, sigma, None


def validate_lambda_factor(
    lambda_default: int, lambda_factor: float, max_large_restarts: int
) -> None:
    """Reject a ``lambda_factor`` whose large-regime λ can drop below 1.

    Large restarts use ``int(lambda_default * lambda_factor**i)`` for
    ``i`` up to ``max_large_restarts``.

    Args:
        lambda_default: First-run offspring count.
        lambda_factor: Population growth factor per large restart.
        max_large_restarts: Largest exponent ``i``.

    Raises:
        ValueError: If any reachable large-regime λ is less than 1.
    """
    exponents = range(max_large_restarts + 1)
    smallest = min((int(lambda_default * lambda_factor**i) for i in exponents), default=1)
    if smallest < 1:
        raise ValueError(
            f"lambda_factor={lambda_factor} with offsprings={lambda_default} and "
            f"max_large_restarts={max_large_restarts} gives a restart λ of {smallest}; "
            "every restart needs at least 1 offspring."
        )


def next_ipop_params(
    lambda_default: int,
    lambda_factor: float,
    irestart_large: int,
    max_large_restarts: int,
    sigma_large: float,
) -> tuple[int, float, int]:
    """Return the next IPOP offspring count, sigma, and large-restart index."""
    irestart_large = min(irestart_large + 1, max_large_restarts)
    lamb = int(lambda_default * lambda_factor**irestart_large)
    return lamb, sigma_large, irestart_large


def next_bipop_params(
    *,
    lambda_default: int,
    lambda_factor: float,
    lambda_large: int,
    irestart_large: int,
    max_large_restarts: int,
    sigma_large: float,
    restart_count: int,
    evals_used: int,
    budget: int,
    budget_large: int,
    budget_small: int,
) -> tuple[int, float, Literal["large", "small"], int, int]:
    """Return the next BIPOP offspring count, sigma, regime, and large-λ state."""
    force_large = restart_count == 1 or evals_used >= budget * 0.95
    if force_large or budget_small >= budget_large:
        irestart_large = min(irestart_large + 1, max_large_restarts)
        lambda_large = int(lambda_default * lambda_factor**irestart_large)
        return lambda_large, sigma_large, "large", irestart_large, lambda_large
    lamb = sample_small_lambda(lambda_default, lambda_large)
    return lamb, sample_small_sigma(sigma_large), "small", irestart_large, lambda_large
