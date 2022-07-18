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
from deap_er.datatypes import Individual
from typing import Optional, Callable
from math import sqrt, exp
import numpy
import copy


__all__ = ['StrategyOnePlusLambda']


# ====================================================================================== #
class StrategyOnePlusLambda:
    """
    A CMA evolution strategy that uses the *lambda* paradigm.

    Parameters:
        parent: A mutable sequence that indicates where to start
            the evolution. The parent requires a fitness attribute.
        sigma: The initial standard deviation of the distribution.
        kwargs: One or more keyword arguments, optional.
    """
    # -------------------------------------------------------- #
    def __init__(self, parent: Individual, sigma: float, **kwargs: Optional):
        if not hasattr(parent, 'fitness'):
            raise TypeError('The parent must have a fitness attribute.')

        self.parent = parent
        self.sigma = sigma

        self.dim = len(self.parent)
        self.C = numpy.identity(self.dim)
        self.A = numpy.identity(self.dim)
        self.pc = numpy.zeros(self.dim)

        self.lambda_ = None
        self.p_thresh = None
        self.damps = None
        self.pt_arg = None
        self.cp = None
        self.c_cum = None
        self.c_cov = None
        self.psucc = None

        self.compute_params(**kwargs)

    # -------------------------------------------------------- #
    def compute_params(self, **kwargs: Optional) -> None:
        """
        Computes the parameters of the strategy based on the *lambda* parameter.
        This function is called automatically when this strategy is instantiated, but
        it needs to be called again if the *lambda* parameter changes during evolution.

        Parameters:
            kwargs: One or more keyword arguments, optional.
        Returns:
            None
        """
        self.lambda_ = kwargs.get("lambda", 1)
        self.p_thresh = kwargs.get("pthresh", 0.44)

        default = 1.0 + self.dim / (2.0 * self.lambda_)
        self.damps = kwargs.get("d", default)

        default = 1.0 / (5 + sqrt(self.lambda_) / 2.0)
        self.pt_arg = kwargs.get("ptarg", default)

        default = self.pt_arg * self.lambda_ / (2 + self.pt_arg * self.lambda_)
        self.cp = kwargs.get("cp", default)

        default = 2.0 / (self.dim + 2.0)
        self.c_cum = kwargs.get("cc", default)

        default = 2.0 / (self.dim ** 2 + 6.0)
        self.c_cov = kwargs.get("ccov", default)

        self.psucc = self.pt_arg

    # -------------------------------------------------------- #
    def generate(self, ind_init: Callable) -> list:
        """
        Generate a population of 'lambda' individuals of
        type *ind_init* from the current strategy.

        Parameters:
            ind_init: A callable object that will be
                used to generate the individuals.
        Returns:
            A list of individuals.
        """
        arz = numpy.random.standard_normal((self.lambda_, self.dim))
        arz = self.parent + self.sigma * numpy.dot(arz, self.A.T)
        return list(map(ind_init, arz))

    # -------------------------------------------------------- #
    def update(self, population: list) -> None:
        """
        Updates the current covariance matrix strategy from the *population*.

        Parameters:
            population: A list of individuals.
        Returns:
            None
        """
        if hasattr(self.parent, 'fitness'):
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            lambda_succ = sum(self.parent.fitness <= ind.fitness for ind in population)
            psucc = float(lambda_succ) / self.lambda_
            self.psucc = (1 - self.cp) * self.psucc + self.cp * psucc

            if self.parent.fitness <= population[0].fitness:
                x_step = (population[0] - numpy.array(self.parent)) / self.sigma
                self.parent = copy.deepcopy(population[0])
                if self.psucc < self.p_thresh:
                    temp_1 = sqrt(self.c_cum * (2 - self.c_cum))
                    self.pc = (1 - self.c_cum) * self.pc + temp_1 * x_step
                    temp_1 = numpy.outer(self.pc, self.pc)
                    self.C = (1 - self.c_cov) * self.C + self.c_cov * temp_1
                else:
                    self.pc = (1 - self.c_cum) * self.pc
                    temp_1 = numpy.outer(self.pc, self.pc)
                    temp_2 = temp_1 + self.c_cum * (2 - self.c_cum) * self.C
                    self.C = (1 - self.c_cov) * self.C + self.c_cov * temp_2

            temp_1 = (self.psucc - self.pt_arg)
            self.sigma *= exp(1.0 / self.damps * temp_1 / (1.0 - self.pt_arg))
            self.A = numpy.linalg.cholesky(self.C)
