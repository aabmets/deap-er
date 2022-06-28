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
from math import sqrt, exp
import numpy
import copy


# ====================================================================================== #
class StrategyOnePlusLambda:

    def __init__(self, parent, sigma, **kwargs):
        self.parent = parent
        self.sigma = sigma
        self.dim = len(self.parent)

        self.C = numpy.identity(self.dim)
        self.A = numpy.identity(self.dim)
        self.pc = numpy.zeros(self.dim)

        self.lambda_ = kwargs.get("lambda_", 1)
        self.d = kwargs.get("d", 1.0 + self.dim / (2.0 * self.lambda_))
        self.pt_arg = kwargs.get("ptarg", 1.0 / (5 + sqrt(self.lambda_) / 2.0))
        self.cp = kwargs.get("cp", self.pt_arg * self.lambda_ / (2 + self.pt_arg * self.lambda_))
        self.cc = kwargs.get("cc", 2.0 / (self.dim + 2.0))
        self.c_cov = kwargs.get("ccov", 2.0 / (self.dim ** 2 + 6.0))
        self.p_thresh = kwargs.get("pthresh", 0.44)

        self.psucc = self.pt_arg

    def generate(self, ind_init):
        arz = numpy.random.standard_normal((self.lambda_, self.dim))
        arz = self.parent + self.sigma * numpy.dot(arz, self.A.T)
        return map(ind_init, arz)

    def update(self, population):
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        lambda_succ = sum(self.parent.fitness <= ind.fitness for ind in population)
        psucc = float(lambda_succ) / self.lambda_
        self.psucc = (1 - self.cp) * self.psucc + self.cp * psucc

        if self.parent.fitness <= population[0].fitness:
            x_step = (population[0] - numpy.array(self.parent)) / self.sigma
            self.parent = copy.deepcopy(population[0])
            if self.psucc < self.p_thresh:
                self.pc = (1 - self.cc) * self.pc + sqrt(self.cc * (2 - self.cc)) * x_step
                self.C = (1 - self.c_cov) * self.C + self.c_cov * numpy.outer(self.pc, self.pc)
            else:
                self.pc = (1 - self.cc) * self.pc
                temp = (numpy.outer(self.pc, self.pc) + self.cc * (2 - self.cc) * self.C)
                self.C = (1 - self.c_cov) * self.C + self.c_cov * temp

        self.sigma = self.sigma * exp(1.0 / self.d * (self.psucc - self.pt_arg) / (1.0 - self.pt_arg))
        self.A = numpy.linalg.cholesky(self.C)
