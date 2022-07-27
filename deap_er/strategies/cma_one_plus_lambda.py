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
from deap_er.base import Individual
from typing import Optional, Callable
from math import sqrt, exp
import numpy
import copy


__all__ = ['StrategyOnePlusLambda']


# ====================================================================================== #
class StrategyOnePlusLambda:
    """
    The one-plus-lambda Covariance Matrix Adaptation evolution strategy.

    :param parent: A mutable sequence that indicates where to start
        the evolution. The parent requires a fitness attribute.
    :param sigma: The initial standard deviation of the distribution.
    :param kwargs: One or more keyword arguments, optional.
    :type parent: :ref:`Individual <datatypes>`

    .. dropdown:: Table of Kwargs
       :margin: 0 5 0 0

       * lambda_ - *(int)*
          * The number of children to produce at each generation.
          * *Default:* :code:`1`
       * ss_dmp - *(float)*
          * Damping of the step-size.
          * *Default:* :code:`1.0 + len(population) / 2.0 * lambda`
       * th_cum - *(float)*
          * Time horizon of the cumulative contribution.
          * *Default:* :code:`2.0 / (len(population) + 2.0)`
       * tgt_sr - *(float)*
          * Target success rate.
          * *Default:* :code:`1.0 / (5 + sqrt(lambda) / 2.0)`
       * thresh_sr - *(float)*
          * Threshold success rate.
          * *Default:* :code:`0.44`
       * ss_learn_rate - *(float)*
          * Learning rate of the step-size.
          * *Default:* :code:`tgt_sr * lambda / (2.0 + tgt_sr * lambda)`
       * cm_learn_rate - *(float)*
          * Learning rate of the covariance matrix.
          * *Default:* :code:`2.0 / (len(population) ** 2 + 6.0)`
    """
    # -------------------------------------------------------- #
    def __init__(self, parent: Individual, sigma: float, **kwargs: Optional):
        if not hasattr(parent, 'fitness'):
            raise TypeError('The parent must have a fitness attribute.')

        self.parent = parent
        self.sigma = sigma

        self.dim = len(self.parent)
        self.big_c = numpy.identity(self.dim)
        self.big_a = numpy.identity(self.dim)
        self.pc = numpy.zeros(self.dim)

        self.lamb = None
        self.thresh_sr = None
        self.ss_dmp = None
        self.tgt_sr = None
        self.ss_learn_rate = None
        self.th_cum = None
        self.cm_learn_rate = None
        self.psucc = None

        self.compute_params(**kwargs)

    # -------------------------------------------------------- #
    def compute_params(self, **kwargs: Optional) -> None:
        """
        Computes the parameters of the strategy based on the *lambda*
        parameter. This function is called automatically when this strategy
        is instantiated, but it needs to be called again with the updated
        **kwargs** if the *lambda* parameter changes during evolution.

        :param kwargs: One or more keyword arguments, optional.
        :return: Nothing.
        """
        self.lamb = kwargs.get("lambda_", 1)
        self.thresh_sr = kwargs.get("thresh_sr", 0.44)

        default = 1.0 + self.dim / (2.0 * self.lamb)
        self.ss_dmp = kwargs.get("ss_dmp", default)

        default = 1.0 / (5 + sqrt(self.lamb) / 2.0)
        self.tgt_sr = kwargs.get("tgt_sr", default)

        default = self.tgt_sr * self.lamb / (2 + self.tgt_sr * self.lamb)
        self.ss_learn_rate = kwargs.get("ss_learn_rate", default)

        default = 2.0 / (self.dim + 2.0)
        self.th_cum = kwargs.get("th_cum", default)

        default = 2.0 / (self.dim ** 2 + 6.0)
        self.cm_learn_rate = kwargs.get("cm_learn_rate", default)

        self.psucc = self.tgt_sr

    # -------------------------------------------------------- #
    def generate(self, ind_init: Callable) -> list:
        """
        Generates a population of *lambda* individuals of
        type **ind_init** from the current strategy.

        :param ind_init: A callable object that generates individuals.
        :return: A list of individuals.
        """
        arz = numpy.random.standard_normal((self.lamb, self.dim))
        arz = self.parent + self.sigma * numpy.dot(arz, self.big_a.T)
        return list(map(ind_init, arz))

    # -------------------------------------------------------- #
    def update(self, population: list) -> None:
        """
        Updates the current CMA strategy from the **population**.

        :param population: A list of individuals.
        :return: Nothing.
        """
        if hasattr(self.parent, 'fitness'):
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            lambda_succ = sum(self.parent.fitness <= ind.fitness for ind in population)
            psucc = float(lambda_succ) / self.lamb
            self.psucc = (1 - self.ss_learn_rate) * self.psucc + self.ss_learn_rate * psucc

            if self.parent.fitness <= population[0].fitness:
                x_step = (population[0] - numpy.array(self.parent)) / self.sigma
                self.parent = copy.deepcopy(population[0])
                if self.psucc < self.thresh_sr:
                    temp_1 = sqrt(self.th_cum * (2 - self.th_cum))
                    self.pc = (1 - self.th_cum) * self.pc + temp_1 * x_step
                    temp_1 = numpy.outer(self.pc, self.pc)
                    self.big_c = (1 - self.cm_learn_rate) * self.big_c + self.cm_learn_rate * temp_1
                else:
                    self.pc = (1 - self.th_cum) * self.pc
                    temp_1 = numpy.outer(self.pc, self.pc)
                    temp_2 = temp_1 + self.th_cum * (2 - self.th_cum) * self.big_c
                    self.big_c = (1 - self.cm_learn_rate) * self.big_c + self.cm_learn_rate * temp_2

            temp_1 = (self.psucc - self.tgt_sr)
            self.sigma *= exp(1.0 / self.ss_dmp * temp_1 / (1.0 - self.tgt_sr))
            self.big_a = numpy.linalg.cholesky(self.big_c)
