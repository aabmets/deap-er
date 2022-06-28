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
        self.chiN = sqrt(self.dim) * (1 - 1. / (4. * self.dim) + 1. / (21. * self.dim ** 2))

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

        self.cc = kwargs.get("ccum", 4. / (self.dim + 4.))
        self.cs = kwargs.get("cs", (self.mu_eff + 2.) / (self.dim + self.mu_eff + 3.))
        self.ccov1 = kwargs.get("ccov1", 2. / ((self.dim + 1.3) ** 2 + self.mu_eff))
        self.ccovmu = kwargs.get("ccovmu", 2. * (self.mu_eff - 2. + 1. / self.mu_eff)
                                 / ((self.dim + 2.) ** 2 + self.mu_eff))
        self.ccovmu = min(1 - self.ccov1, self.ccovmu)
        self.damps = 1. + 2. * max(0., sqrt((self.mu_eff - 1.) / (self.dim + 1.)) - 1.) + self.cs
        self.damps = kwargs.get("damps", self.damps)

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

        self.ps = (1 - self.cs) * self.ps + sqrt(self.cs * (2 - self.cs) * self.mu_eff) / \
            self.sigma * numpy.dot(self.B, (1. / self.diagD) * numpy.dot(self.B.T, c_diff))

        temp = sqrt(1. - (1. - self.cs) ** (2. * (self.update_count + 1.)))
        temp = numpy.linalg.norm(self.ps) / temp / self.chiN < (1.4 + 2. / (self.dim + 1.))
        hsig = float(temp)

        temp = sqrt(self.cc * (2 - self.cc) * self.mu_eff)
        self.pc = (1 - self.cc) * self.pc + hsig * temp / self.sigma * c_diff

        ar_tmp = population[0:self.mu] - old_centroid
        self.C = (1 - self.ccov1 - self.ccovmu + (1 - hsig) * self.ccov1 * self.cc * (2 - self.cc)) * \
            self.C + self.ccov1 * numpy.outer(self.pc, self.pc) + self.ccovmu * \
            numpy.dot((self.weights * ar_tmp.T), ar_tmp) / self.sigma ** 2

        self.sigma *= numpy.exp((numpy.linalg.norm(self.ps) / self.chiN - 1.) * self.cs / self.damps)

        self.diagD, self.B = numpy.linalg.eigh(self.C)
        indx = numpy.argsort(self.diagD)

        self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]]

        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD

        self.update_count += 1
