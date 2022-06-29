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
from math import sqrt, log
import numpy


# ====================================================================================== #
class Strategy:

    def __init__(self, centroid, sigma, **kwargs):
        """
        A strategy that will keep track of the basic parameters of the CMA-ES algorithm.

        :param centroid: An iterable object that indicates where to start the evolution.
        :param sigma: The initial standard deviation of the distribution.
        :param kwargs: One or more optional keyword arguments described in the documentation.
        """
        self.update_count = 0
        self.centroid = numpy.array(centroid)

        self.dim = len(self.centroid)
        self.sigma = sigma
        self.pc = numpy.zeros(self.dim)
        self.ps = numpy.zeros(self.dim)

        temp_1 = 1 - 1. / (4. * self.dim) + 1. / (21. * self.dim ** 2)
        self.chiN = sqrt(self.dim) * temp_1

        self.C = kwargs.get("cmatrix", numpy.identity(self.dim))
        self.diagD, self.B = numpy.linalg.eigh(self.C)

        indx = numpy.argsort(self.diagD)
        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD

        self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]]
        self.lambda_ = kwargs.get("lambda_", int(4 + 3 * log(self.dim)))
        self.mu = kwargs.get("mu", int(self.lambda_ / 2))

        r_weights = kwargs.get("weights", "superlinear")
        if r_weights == "superlinear":
            self.weights = log(self.mu + 0.5) - numpy.log(numpy.arange(1, self.mu + 1))
        elif r_weights == "linear":
            self.weights = self.mu + 0.5 - numpy.arange(1, self.mu + 1)
        elif r_weights == "equal":
            self.weights = numpy.ones(self.mu)
        else:
            raise RuntimeError("Unknown weights : %s" % r_weights)

        self.weights /= sum(self.weights)
        self.mu_eff = 1. / sum(self.weights ** 2)

        default = 4. / (self.dim + 4.)
        self.c_cum = kwargs.get("ccum", default)

        default = (self.mu_eff + 2.) / (self.dim + self.mu_eff + 3.)
        self.cs = kwargs.get("cs", default)

        default = 2. / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.c_cov_1 = kwargs.get("ccov1", default)

        temp_1 = self.mu_eff - 2. + 1. / self.mu_eff
        temp_2 = (self.dim + 2.) ** 2 + self.mu_eff
        default = 2. * temp_1 / temp_2
        self.c_cov_mu = kwargs.get("ccovmu", default)
        self.c_cov_mu = min(1 - self.c_cov_1, self.c_cov_mu)

        temp_1 = sqrt((self.mu_eff - 1.) / (self.dim + 1.))
        temp_2 = max(0., temp_1 - 1.)
        default = 1. + 2. * temp_2 + self.cs
        self.damps = kwargs.get("damps", default)

    # -------------------------------------------------------------------------------------- #
    def generate(self, ind_init):
        arz = numpy.random.standard_normal((self.lambda_, self.dim))
        arz = self.centroid + self.sigma * numpy.dot(arz, self.BD.T)
        return map(ind_init, arz)

    # -------------------------------------------------------------------------------------- #
    def update(self, population):
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        old_centroid = self.centroid
        self.centroid = numpy.dot(self.weights, population[0:self.mu])

        c_diff = self.centroid - old_centroid

        temp_1 = sqrt(self.cs * (2 - self.cs) * self.mu_eff)
        temp_2 = numpy.dot(self.B, (1. / self.diagD) * numpy.dot(self.B.T, c_diff))
        self.ps = (1 - self.cs) * self.ps + temp_1 / self.sigma * temp_2

        temp_1 = sqrt(1. - (1. - self.cs) ** (2. * (self.update_count + 1.)))
        temp_2 = numpy.linalg.norm(self.ps) / temp_1 / self.chiN < (1.4 + 2. / (self.dim + 1.))
        hsig = float(temp_2)

        temp_1 = sqrt(self.c_cum * (2 - self.c_cum) * self.mu_eff)
        self.pc = (1 - self.c_cum) * self.pc + hsig * temp_1 / self.sigma * c_diff

        ar_tmp = population[0:self.mu] - old_centroid
        temp_0 = (1 - hsig) * self.c_cov_1 * self.c_cum * (2 - self.c_cum)
        temp_1 = 1 - self.c_cov_1 - self.c_cov_mu + temp_0
        temp_2 = numpy.outer(self.pc, self.pc)
        temp_3 = numpy.dot((self.weights * ar_tmp.T), ar_tmp)
        self.C = temp_1 * self.C + self.c_cov_1 * temp_2 + self.c_cov_mu * temp_3 / self.sigma ** 2

        temp = (numpy.linalg.norm(self.ps) / self.chiN - 1.)
        self.sigma *= numpy.exp(temp * self.cs / self.damps)

        self.diagD, self.B = numpy.linalg.eigh(self.C)
        indx = numpy.argsort(self.diagD)

        self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]]

        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD

        self.update_count += 1
