# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 - Mattias Aabmets, The DEAP Team and Other Contributors           #
#                                                                                        #
#   Permission is hereby granted, free of charge, to any person obtaining a copy         #
#   of this software and associated documentation files (the "Software"), to deal        #
#   in the Software without restriction, including without limitation the rights         #
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell            #
#   copies of the Software, and to permit persons to whom the Software is                #
#   furnished to do so, subject to the following conditions:                             #
#                                                                                        #
#   The above copyright notice and this permission notice shall be included in all       #
#   copies or substantial portions of the Software.                                      #
#                                                                                        #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR           #
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,             #
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE          #
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER               #
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,        #
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE        #
#   SOFTWARE.                                                                            #
#                                                                                        #
# ====================================================================================== #
"""
A module that provides support for the Covariance Matrix Adaptation Evolution Strategy.
"""
from deap_er import tools
from math import sqrt, log, exp
import numpy
import copy


# ====================================================================================== #
class StrategyMultiObjective(object):
    """
    Multiobjective CMA-ES strategy based on the paper [Voss2010]_. It
    is used similarly as the standard CMA-ES strategy with a generate-update
    scheme.

    :param population: An initial population of individual.
    :param sigma: The initial step size of the complete system.
    :param mu: The number of parents to use in the evolution. When not
               provided it defaults to the length of *population*. (optional)
    :param lambda_: The number of offspring to produce at each generation.
                    (optional, defaults to 1)
    :param indicator: The indicator function to use. (optional, default to
                      :func:`~deap.tools.hypervolume`)

    Other parameters can be provided as described in the next table

    +----------------+---------------------------+----------------------------+
    | Parameter      | Default                   | Details                    |
    +================+===========================+============================+
    | ``d``          | ``1.0 + N / 2.0``         | Damping for step-size.     |
    +----------------+---------------------------+----------------------------+
    | ``ptarg``      | ``1.0 / (5 + 1.0 / 2.0)`` | Target success rate.        |
    +----------------+---------------------------+----------------------------+
    | ``cp``         | ``ptarg / (2.0 + ptarg)`` | Step size learning rate.   |
    +----------------+---------------------------+----------------------------+
    | ``cc``         | ``2.0 / (N + 2.0)``       | Cumulation time horizon.   |
    +----------------+---------------------------+----------------------------+
    | ``ccov``       | ``2.0 / (N**2 + 6.0)``    | Covariance matrix learning |
    |                |                           | rate.                      |
    +----------------+---------------------------+----------------------------+
    | ``pthresh``    | ``0.44``                  | Threshold success rate.    |
    +----------------+---------------------------+----------------------------+

    .. [Voss2010] Voss, Hansen, Igel, "Improved Step Size Adaptation
       for the MO-CMA-ES", 2010.

    """

    # -------------------------------------------------------------------------------------- #
    def __init__(self, population, sigma, **params):
        self.parents = population
        self.dim = len(self.parents[0])

        # Selection
        self.mu = params.get("mu", len(self.parents))
        self.lambda_ = params.get("lambda_", 1)

        # Step size control
        self.d = params.get("d", 1.0 + self.dim / 2.0)
        self.ptarg = params.get("ptarg", 1.0 / (5.0 + 0.5))
        self.cp = params.get("cp", self.ptarg / (2.0 + self.ptarg))

        # Covariance matrix adaptation
        self.cc = params.get("cc", 2.0 / (self.dim + 2.0))
        self.ccov = params.get("ccov", 2.0 / (self.dim ** 2 + 6.0))
        self.pthresh = params.get("pthresh", 0.44)

        # Internal parameters associated to the mu parent
        self.sigmas = [sigma] * len(population)
        # Lower Cholesky matrix (Sampling matrix)
        self.A = [numpy.identity(self.dim) for _ in range(len(population))]
        # Inverse Cholesky matrix (Used in the update of A)
        self.invCholesky = [numpy.identity(self.dim) for _ in range(len(population))]
        self.pc = [numpy.zeros(self.dim) for _ in range(len(population))]
        self.psucc = [self.ptarg] * len(population)

        self.indicator = params.get("indicator", tools.hypervolume)

    # -------------------------------------------------------------------------------------- #
    def generate(self, ind_init):
        """Generate a population of :math:`\lambda` individuals of type
        *ind_init* from the current strategy.

        :param ind_init: A function object that is able to initialize an
                         individual from a list.
        :returns: A list of individuals with a private attribute :attr:`_ps`.
                  This last attribute is essential to the update function, it
                  indicates that the individual is an offspring and the index
                  of its parent.
        """
        arz = numpy.random.randn(self.lambda_, self.dim)
        individuals = list()

        # Make sure every parent has a parent tag and index
        for i, p in enumerate(self.parents):
            p._ps = "p", i

        # Each parent produce an offspring
        if self.lambda_ == self.mu:
            for i in range(self.lambda_):
                # print "Z", list(arz[i])
                individuals.append(ind_init(self.parents[i] + self.sigmas[i] * numpy.dot(self.A[i], arz[i])))
                individuals[-1]._ps = "o", i

        # Parents producing an offspring are chosen at random from the first front
        else:
            ndom = tools.sortLogNondominated(self.parents, len(self.parents), first_front_only=True)
            for i in range(self.lambda_):
                j = numpy.random.randint(0, len(ndom))
                _, p_idx = ndom[j]._ps
                individuals.append(ind_init(self.parents[p_idx] + self.sigmas[p_idx] * numpy.dot(self.A[p_idx], arz[i])))
                individuals[-1]._ps = "o", p_idx

        return individuals

    # -------------------------------------------------------------------------------------- #
    def _select(self, candidates):
        if len(candidates) <= self.mu:
            return candidates, []

        pareto_fronts = tools.sortLogNondominated(candidates, len(candidates))

        chosen = list()
        mid_front = None
        not_chosen = list()

        full = False
        for front in pareto_fronts:
            if len(chosen) + len(front) <= self.mu and not full:
                chosen += front
            elif mid_front is None and len(chosen) < self.mu:
                mid_front = front
                full = True
            else:
                not_chosen += front

        k = self.mu - len(chosen)
        if k > 0:
            ref_point = numpy.array([ind.fitness.wvalues for ind in candidates]) * -1
            ref_point = numpy.max(ref_point, axis=0) + 1

            for _ in range(len(mid_front) - k):
                idx = self.indicator(mid_front, ref=ref_point)
                not_chosen.append(mid_front.pop(idx))

            chosen += mid_front

        return chosen, not_chosen

    # -------------------------------------------------------------------------------------- #
    def _rankOneUpdate(self, invCholesky, A, alpha, beta, v):
        w = numpy.dot(invCholesky, v)

        # Under this threshold, the update is mostly noise
        if w.max() > 1e-20:
            w_inv = numpy.dot(w, invCholesky)
            norm_w2 = numpy.sum(w ** 2)
            a = sqrt(alpha)
            root = numpy.sqrt(1 + beta / alpha * norm_w2)
            b = a / norm_w2 * (root - 1)

            A = a * A + b * numpy.outer(v, w)
            invCholesky = 1.0 / a * invCholesky - b / (a ** 2 + a * b * norm_w2) * numpy.outer(w, w_inv)

        return invCholesky, A

    # -------------------------------------------------------------------------------------- #
    def update(self, population):
        """Update the current covariance matrix strategies from the
        *population*.

        :param population: A list of individuals from which to update the
                           parameters.
        """
        chosen, not_chosen = self._select(population + self.parents)

        cp, cc, ccov = self.cp, self.cc, self.ccov
        d, ptarg, pthresh = self.d, self.ptarg, self.pthresh

        # Make copies for chosen offspring only
        last_steps = [self.sigmas[ind._ps[1]] if ind._ps[0] == "o" else None for ind in chosen]
        sigmas = [self.sigmas[ind._ps[1]] if ind._ps[0] == "o" else None for ind in chosen]
        invCholesky = [self.invCholesky[ind._ps[1]].copy() if ind._ps[0] == "o" else None for ind in chosen]
        A = [self.A[ind._ps[1]].copy() if ind._ps[0] == "o" else None for ind in chosen]
        pc = [self.pc[ind._ps[1]].copy() if ind._ps[0] == "o" else None for ind in chosen]
        psucc = [self.psucc[ind._ps[1]] if ind._ps[0] == "o" else None for ind in chosen]

        # Update the internal parameters for successful offspring
        for i, ind in enumerate(chosen):
            t, p_idx = ind._ps

            # Only the offspring update the parameter set
            if t == "o":
                # Update (Success = 1 since it is chosen)
                psucc[i] = (1.0 - cp) * psucc[i] + cp
                sigmas[i] = sigmas[i] * exp((psucc[i] - ptarg) / (d * (1.0 - ptarg)))

                if psucc[i] < pthresh:
                    xp = numpy.array(ind)
                    x = numpy.array(self.parents[p_idx])
                    pc[i] = (1.0 - cc) * pc[i] + sqrt(cc * (2.0 - cc)) * (xp - x) / last_steps[i]
                    invCholesky[i], A[i] = self._rankOneUpdate(invCholesky[i], A[i], 1 - ccov, ccov, pc[i])
                else:
                    pc[i] = (1.0 - cc) * pc[i]
                    pc_weight = cc * (2.0 - cc)
                    invCholesky[i], A[i] = self._rankOneUpdate(invCholesky[i], A[i], 1 - ccov + pc_weight, ccov, pc[i])

                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx] + cp
                self.sigmas[p_idx] = self.sigmas[p_idx] * exp((self.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg)))

        # It is unnecessary to update the entire parameter set for not chosen individuals
        # Their parameters will not make it to the next generation
        for ind in not_chosen:
            t, p_idx = ind._ps

            # Only the offspring update the parameter set
            if t == "o":
                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx]
                self.sigmas[p_idx] = self.sigmas[p_idx] * exp((self.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg)))

        # Make a copy of the internal parameters
        # The parameter is in the temporary variable for offspring and in the original one for parents
        self.parents = chosen
        self.sigmas = [sigmas[i] if ind._ps[0] == "o" else self.sigmas[ind._ps[1]] for i, ind in enumerate(chosen)]
        self.invCholesky = [invCholesky[i] if ind._ps[0] == "o" else self.invCholesky[ind._ps[1]] for i, ind in enumerate(chosen)]
        self.A = [A[i] if ind._ps[0] == "o" else self.A[ind._ps[1]] for i, ind in enumerate(chosen)]
        self.pc = [pc[i] if ind._ps[0] == "o" else self.pc[ind._ps[1]] for i, ind in enumerate(chosen)]
        self.psucc = [psucc[i] if ind._ps[0] == "o" else self.psucc[ind._ps[1]] for i, ind in enumerate(chosen)]