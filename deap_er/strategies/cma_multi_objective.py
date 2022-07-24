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
from deap_er import tools
from typing import Optional, Callable
from math import sqrt, exp
import numpy


__all__ = ['StrategyMultiObjective']


# ====================================================================================== #
class StrategyMultiObjective:
    """
    The multi-objective Covariance Matrix Adaptation evolution strategy.

    :param population: An initial population of individuals.
    :param sigma: The initial step size of the complete system.
    :param kwargs: One or more keyword arguments, optional.

    .. dropdown:: Table of Kwargs
       :margin: 0 5 0 0

       * lambda - *(int)*
          * The number of children to produce at each generation.
          * *Default:* :code:`1`
       * mu - *(int)*
          * The number of children to keep as parents for the next generation.
          * *Default:* :code:`len(population)`
       * ss_dmp - *(float)*
          * Damping of the step-size.
          * *Default:* :code:`1.0 + len(population) / 2.0`
       * th_cum - *(float)*
          * Time horizon of the cumulative contribution.
          * *Default:* :code:`2.0 / (len(population) + 2.0)`
       * tgt_sr - *(float)*
          * Target success rate.
          * *Default:* :code:`1.0 / 5.5`
       * thresh_sr - *(float)*
          * Threshold success rate.
          * *Default:* :code:`0.44`
       * ss_learn_rate - *(float)*
          * Learning rate of the step-size.
          * *Default:* :code:`tgt_sr / (2.0 + tgt_sr)`
       * cm_learn_rate - *(float)*
          * Learning rate of the covariance matrix.
          * *Default:* :code:`2.0 / (len(population) ** 2 + 6.0)`
       * indicator - *(Callable)*
          * The indicator function to use.
          * *Default:* :code:`deap_er.utilities.least_contrib`
    """
    # -------------------------------------------------------- #
    def __init__(self, population: list, sigma: float, **kwargs: Optional):
        self.parents = population
        self.dim = len(self.parents[0])
        pop_size = len(population)

        self.mu = kwargs.get("mu", pop_size)
        self.lamb = kwargs.get("lambda", 1)
        self.ss_dmp = kwargs.get("ss_dmp", 1.0 + self.dim / 2.0)
        self.tgt_sr = kwargs.get("tgt_sr", 1.0 / (5.0 + 0.5))
        self.ss_learn_rate = kwargs.get("ss_learn_rate", self.tgt_sr / (2.0 + self.tgt_sr))
        self.th_cum = kwargs.get("th_cum", 2.0 / (self.dim + 2.0))
        self.cm_learn_rate = kwargs.get("cm_learn_rate", 2.0 / (self.dim ** 2 + 6.0))
        self.thresh_sr = kwargs.get("thresh_sr", 0.44)
        self.indicator = kwargs.get("indicator", tools.least_contrib)
        self.timeout = kwargs.get("timeout", 60)

        self.sigmas = [sigma] * pop_size
        self.big_a = [numpy.identity(self.dim) for _ in range(pop_size)]
        self.inv_cholesky = [numpy.identity(self.dim) for _ in range(pop_size)]
        self.pc = [numpy.zeros(self.dim) for _ in range(pop_size)]
        self.psucc = [self.tgt_sr] * pop_size

    # -------------------------------------------------------- #
    def _select(self, candidates):
        if len(candidates) <= self.mu:
            return candidates, []

        pareto_fronts = tools.sort_log_non_dominated(candidates, len(candidates))

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
            ref = [ind.fitness.wvalues for ind in candidates]
            ref = numpy.array(ref) * -1
            ref = numpy.max(ref, axis=0) + 1

            for _ in range(len(mid_front) - k):
                idx = self.indicator(mid_front, ref, self.timeout)
                not_chosen.append(mid_front.pop(idx))

            chosen += mid_front

        return chosen, not_chosen

    # -------------------------------------------------------- #
    @staticmethod
    def _rank_one_update(inv_cholesky, big_a, alpha, beta, v):
        w = numpy.dot(inv_cholesky, v)

        if w.max(initial=None) > 1e-20:
            w_inv = numpy.dot(w, inv_cholesky)
            norm_w2 = numpy.sum(w ** 2)
            a = sqrt(alpha)
            root = numpy.sqrt(1 + beta / alpha * norm_w2)
            b = a / norm_w2 * (root - 1)
            big_a = a * big_a + b * numpy.outer(v, w)
            part = (a ** 2 + a * b * norm_w2)
            inv_cholesky = 1.0 / a * inv_cholesky - b / part * numpy.outer(w, w_inv)

        return inv_cholesky, big_a

    # -------------------------------------------------------- #
    def update(self, population: list) -> None:
        """
        Updates the current CMA strategy from the **population**.

        :param population: A list of individuals.
        :return: Nothing.
        """
        chosen, not_chosen = self._select(population + self.parents)

        cp, cc, c_cov = self.ss_learn_rate, self.th_cum, self.cm_learn_rate
        d, pt_arg, p_thresh = self.ss_dmp, self.tgt_sr, self.thresh_sr

        bag = [list() for _ in range(6)]
        for ind in chosen:
            if ind.ps_[0] == 'o':
                idx = ind.ps_[1]
                bag[0].append(self.sigmas[idx])
                bag[1].append(self.sigmas[idx])
                bag[2].append(self.inv_cholesky[idx].copy())
                bag[3].append(self.big_a[idx].copy())
                bag[4].append(self.pc[idx].copy())
                bag[5].append(self.psucc[idx])
            else:
                for b in bag:
                    b.append(None)

        last_steps, sigmas, inv_cholesky = bag[0], bag[1], bag[2]
        big_a, pc, psucc = bag[3], bag[4], bag[5]

        for i, ind in enumerate(chosen):
            t, p_idx = ind.ps_

            if t == "o":
                psucc[i] = (1.0 - cp) * psucc[i] + cp
                sigmas[i] = sigmas[i] * exp((psucc[i] - pt_arg) / (d * (1.0 - pt_arg)))

                if psucc[i] < p_thresh:
                    xp = numpy.array(ind)
                    x = numpy.array(self.parents[p_idx])
                    pc[i] = (1.0 - cc) * pc[i] + sqrt(cc * (2.0 - cc)) * (xp - x) / last_steps[i]
                    inv_cholesky[i], big_a[i] = self._rank_one_update(
                        inv_cholesky[i], big_a[i], 1 - c_cov, c_cov, pc[i]
                    )
                else:
                    pc[i] = (1.0 - cc) * pc[i]
                    pc_weight = cc * (2.0 - cc)
                    inv_cholesky[i], big_a[i] = self._rank_one_update(
                        inv_cholesky[i], big_a[i], 1 - c_cov + pc_weight, c_cov, pc[i]
                    )

                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx] + cp
                exp_ = exp((self.psucc[p_idx] - pt_arg) / (d * (1.0 - pt_arg)))
                self.sigmas[p_idx] = self.sigmas[p_idx] * exp_

        for ind in not_chosen:
            t, p_idx = ind.ps_

            if t == "o":
                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx]
                exp_ = exp((self.psucc[p_idx] - pt_arg) / (d * (1.0 - pt_arg)))
                self.sigmas[p_idx] = self.sigmas[p_idx] * exp_

        sources = {
            'inv_cholesky': inv_cholesky,
            'sigmas': sigmas,
            'big_a': big_a,
            'psucc': psucc,
            'pc': pc
        }
        for name, var in sources.items():
            attr = getattr(self, name)
            bag = list()
            for i, ind in enumerate(chosen):
                if ind.ps_[0] == "o":
                    bag.append(var[i])
                else:
                    bag.append(attr[ind.ps_[1]])
            setattr(self, name, bag)

        self.parents = chosen

    # -------------------------------------------------------- #
    def generate(self, ind_init: Callable) -> list:
        """
        Generates a population of *lambda* individuals of
        type **ind_init** from the current strategy.

        :param ind_init: A callable object that generates individuals.
        :return: A list of individuals.
        """
        arz = numpy.random.randn(self.lamb, self.dim)
        individuals = list()

        for i, p in enumerate(self.parents):
            p.ps_ = "p", i

        if self.lamb == self.mu:
            for i in range(self.lamb):
                dot = numpy.dot(self.big_a[i], arz[i])
                init = ind_init(self.parents[i] + self.sigmas[i] * dot)
                individuals.append(init)
                individuals[-1].ps_ = "o", i

        else:
            n_dom = tools.sort_log_non_dominated(
                self.parents, len(self.parents),
                first_front_only=True)

            for i in range(self.lamb):
                j = numpy.random.randint(0, len(n_dom))
                _, p_idx = n_dom[j].ps_
                dot = numpy.dot(self.big_a[p_idx], arz[i])
                init = ind_init(self.parents[p_idx] + self.sigmas[p_idx] * dot)
                individuals.append(init)
                individuals[-1].ps_ = "o", p_idx

        return individuals
