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
from deap_er.base.dtypes import *
from math import sin, cos, pi, exp, e, sqrt
from functools import reduce
from operator import mul
import random
import numpy


__all__ = [
    'bm_rand', 'bm_plane', 'bm_sphere', 'bm_cigar', 'bm_rosenbrock',
    'bm_h1', 'bm_ackley', 'bm_bohachevsky', 'bm_griewank', 'bm_schaffer',
    'bm_schwefel', 'bm_himmelblau', 'bm_rastrigin', 'bm_rastrigin_scaled',
    'bm_rastrigin_skewed', 'bm_shekel'
]


# ====================================================================================== #
def bm_rand(*_) -> tuple[float]:
    """
    Random test objective function. The unnamed **args** parameter is an input
    sink for DEAP-er internal functionality and has no effect on the result.

    :return: A completely random number.

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization or maximization
          * - Range
            - none
          * - Global optima
            - none
          * - Function
            - :math:`f(\\mathbf{x}) = \\text{random}(0,1)`
    """
    result = random.random()
    return result,


# -------------------------------------------------------------------------------------- #
def bm_plane(individual: Individual) -> tuple[float]:
    """
    Plane test objective function.

    :param individual: The Individual to be evaluated.
    :return: The first Fitness value of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - none
          * - Global optima
            - :math:`x_i = 0, \\forall i \\in \\lbrace 1 \\ldots \
               N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
          * - Function
            - :math:`f(\\mathbf{x}) = x_0`
    """
    result = individual[0]
    return result,


# -------------------------------------------------------------------------------------- #
def bm_sphere(individual: Individual) -> tuple[float]:
    """
    Sphere test objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness value of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - none
          * - Global optima
            - :math:`x_i = 0, \\forall i \\in \\lbrace 1 \\ldots \
               N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
          * - Function
            - :math:`f(\\mathbf{x}) = \\sum_{i=1}^Nx_i^2`
    """
    result = sum(gene * gene for gene in individual)
    return result,


# -------------------------------------------------------------------------------------- #
def bm_cigar(individual: Individual) -> tuple[float]:
    """
    Cigar test objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness value of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - none
          * - Global optima
            - :math:`x_i = 0, \\forall i \\in \\lbrace 1 \\ldots \
               N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
          * - Function
            - :math:`f(\\mathbf{x}) = x_0^2 + 10^6\\sum_{i=1}^N\\,x_i^2`
    """
    _sum = sum(gene * gene for gene in individual[1:])
    result = individual[0] ** 2 + 1e6 * _sum
    return result,


# -------------------------------------------------------------------------------------- #
def bm_rosenbrock(individual: Individual) -> tuple[float]:
    """
    Rosenbrock test objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness value of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - none
          * - Global optima
            - :math:`x_i = 1, \\forall i \\in \\lbrace 1 \\ldots \
               N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
          * - Function
            - :math:`f(\\mathbf{x}) = \\sum_{i=1}^{N-1} \
               (1-x_i)^2 + 100 (x_{i+1} - x_i^2 )^2`
    """
    results = []
    for x, y in zip(individual[:-1], individual[1:]):
        results.append(100 * (x * x - y) ** 2 + (1 - x) ** 2)
    result = sum(results)
    return result,


# -------------------------------------------------------------------------------------- #
def bm_h1(individual: Individual) -> tuple[float]:
    """
    Simple two-dimensional function containing several local maxima.

    :param individual: The Individual to be evaluated.
    :return: Fitness value of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - maximization
          * - Range
            - :math:`x_i \\in [-100, 100]`
          * - Global optima
            - :math:`\\mathbf{x} = (8.6998, 6.7665)`, :math:`f(\\mathbf{x}) = 2`\n
          * - Function
            - :math:`f(\\mathbf{x}) = \\frac{\\sin(x_1 - \\frac{x_2}{8})^2 + \
               \\sin(x_2 + \\frac{x_1}{8})^2}{\\sqrt{(x_1 - 8.6998)^2 + \
               (x_2 - 6.7665)^2} + 1}`
    """
    def compute_num():
        var_1 = sin(individual[0] - individual[1] / 8) ** 2
        var_2 = sin(individual[1] + individual[0] / 8) ** 2
        return var_1 + var_2

    def compute_denum():
        var_1 = (individual[0] - 8.6998) ** 2
        var_2 = (individual[1] - 6.7665) ** 2
        return (var_1 + var_2) ** 0.5 + 1

    result = compute_num() / compute_denum()
    return result,


# -------------------------------------------------------------------------------------- #
def bm_ackley(individual: Individual) -> tuple[float]:
    """
    Ackley test objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness value of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - :math:`x_i \\in [-15, 30]`
          * - Global optima
            - :math:`x_i = 0, \\forall i \\in \\lbrace 1 \\ldots \
               N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
          * - Function
            - :math:`f(\\mathbf{x}) = 20 - 20\\exp\\left(-0.2 \
               \\sqrt{\\frac{1}{N} \\sum_{i=1}^N x_i^2} \
               \\right) + e - \\exp\\left(\\frac{1}{N} \
               \\sum_{i=1}^N \\cos(2\\pi x_i) \\right)`
    """
    len_ind = len(individual)
    exp_1 = exp(-0.2 * sqrt(1 / len_ind * sum(x ** 2 for x in individual)))
    exp_2 = exp(1 / len_ind * sum(cos(2 * pi * x) for x in individual))
    result = 20 - 20 * exp_1 + e - exp_2
    return result,


# -------------------------------------------------------------------------------------- #
def bm_bohachevsky(individual: Individual) -> tuple[float]:
    """
    Bohachevsky test objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness value of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - :math:`x_i \\in [-100, 100]`
          * - Global optima
            - :math:`x_i = 0, \\forall i \\in \\lbrace 1 \\ldots \
               N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
          * - Function
            - :math:`f(\\mathbf{x}) = \\sum_{i=1}^{N-1}(x_i^2 + \
               2x_{i+1}^2 - 0.3\\cos(3\\pi x_i) - 0.4\\cos(4 \
               \\pi x_{i+1}) + 0.7)`
    """
    results = []
    for x, x1 in zip(individual[:-1], individual[1:]):
        c1 = cos(3 * pi * x)
        c2 = cos(4 * pi * x1)
        res = x ** 2 + 2 * x1 ** 2 - 0.3 * c1 - 0.4 * c2 + 0.7
        results.append(res)
    result = sum(results)
    return result,


# -------------------------------------------------------------------------------------- #
def bm_griewank(individual: Individual) -> tuple[float]:
    """
    Griewank test objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness value of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - :math:`x_i \\in [-600, 600]`
          * - Global optima
            - :math:`x_i = 0, \\forall i \\in \\lbrace 1 \\ldots \
               N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
          * - Function
            - :math:`f(\\mathbf{x}) = \\frac{1}{4000}\\sum_{i=1}^N \
               \\,x_i^2 - \\prod_{i=1}^N\\cos\\left( \
               \\frac{x_i}{\\sqrt{i}}\\right) + 1`
    """
    values = [cos(x/sqrt(i+1.0)) for i, x in enumerate(individual)]
    exp_sum = sum(x**2 for x in individual)
    result = 1 / 4000 * exp_sum - reduce(mul, values, 1) + 1
    return result,


# -------------------------------------------------------------------------------------- #
def bm_schaffer(individual: Individual) -> tuple[float]:
    """
    Schaffer test objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness value of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - :math:`x_i \\in [-100, 100]`
          * - Global optima
            - :math:`x_i = 0, \\forall i \\in \\lbrace 1 \
               \\ldots N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
          * - Function
            - :math:`f(\\mathbf{x}) = \\sum_{i=1}^{N-1} \
               (x_i^2+x_{i+1}^2)^{0.25} \\cdot \\left[ \
               \\sin^2(50\\cdot(x_i^2+x_{i+1}^2)^{0.10}) \
               + 1.0 \\right]`
    """
    results = []
    for x, x1 in zip(individual[:-1], individual[1:]):
        var_1 = (x ** 2 + x1 ** 2) ** 0.25
        var_2 = sin(50 * (x ** 2 + x1 ** 2) ** 0.1) ** 2 + 1.0
        results.append(var_1 * var_2)
    result = sum(results)
    return result,


# -------------------------------------------------------------------------------------- #
def bm_schwefel(individual: Individual) -> tuple[float]:
    """
    Schwefel test objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness value of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - :math:`x_i \\in [-500, 500]`
          * - Global optima
            - :math:`x_i = 420.96874636, \\forall i \\in \\lbrace 1 \
               \\ldots N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
          * - Function
            - :math:`f(\\mathbf{x}) = 418.9828872724339\\cdot N - \
               \\sum_{i=1}^N\\,x_i\\sin\\left(\\sqrt{|x_i|}\\right)`
    """
    len_ind = len(individual)
    values = sum(x * sin(sqrt(abs(x))) for x in individual)
    result = 418.9828872724339 * len_ind - values
    return result,


# -------------------------------------------------------------------------------------- #
def bm_himmelblau(individual: Individual) -> tuple[float]:
    """
    The Himmelblaus function has 4 defined
    minimums in the range of :math:`[-6, 6]^2`.

    :param individual: The Individual to be evaluated.
    :return: Fitness value of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - :math:`x_i \\in [-6, 6]`
          * - Global optima
            - :math:`\\mathbf{x}_1 = (3.0, 2.0)`, :math:`f(\\mathbf{x}_1) = 0`\n
              :math:`\\mathbf{x}_2 = (-2.805118, 3.131312)`, :math:`f(\\mathbf{x}_2) = 0`\n
              :math:`\\mathbf{x}_3 = (-3.779310, -3.283186)`, :math:`f(\\mathbf{x}_3) = 0`\n
              :math:`\\mathbf{x}_4 = (3.584428, -1.848126)`, :math:`f(\\mathbf{x}_4) = 0`\n
          * - Function
            - :math:`f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 -7)^2`
    """
    var_1 = (individual[0] * individual[0] + individual[1] - 11) ** 2
    var_2 = (individual[0] + individual[1] * individual[1] - 7) ** 2
    result = var_1 + var_2
    return result,


# -------------------------------------------------------------------------------------- #
def bm_rastrigin(individual: Individual) -> tuple[float]:
    """
    Rastrigin test objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness value of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - :math:`x_i \\in [-5.12, 5.12]`
          * - Global optima
            - :math:`x_i = 0, \\forall i \\in \\lbrace 1 \
               \\ldots N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
          * - Function
            - :math:`f(\\mathbf{x}) = 10N + \\sum_{i=1}^N \
               x_i^2 - 10 \\cos(2\\pi x_i)`
    """
    values = [gene * gene - 10 * cos(2 * pi * gene) for gene in individual]
    result = 10 * len(individual) + sum(values)
    return result,


# -------------------------------------------------------------------------------------- #
def bm_rastrigin_scaled(individual: Individual) -> tuple[float]:
    """
    Scaled Rastrigin test objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness value of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - :math:`x_i \\in [-5.12, 5.12]`
          * - Global optima
            - :math:`x_i = 0, \\forall i \\in \\lbrace 1 \
               \\ldots N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
          * - Function
            - :math:`f(\\mathbf{x}) = 10N + \\sum_{i=1}^N \
               \\left(10^{\\left(\\frac{i-1}{N-1}\\right)} \
               x_i \\right)^2 - 10\\cos\\left(2\\pi 10^{\\left( \
               \\frac{i-1}{N-1}\\right)} x_i \\right)`
    """
    results = []
    len_ind = len(individual)
    for i, x in enumerate(individual):
        var_1 = (10 ** (i / (len_ind - 1)) * x) ** 2
        var_2 = 10 * cos(2 * pi * 10 ** (i / (len_ind - 1)) * x)
        results.append(var_1 - var_2)
    result = 10 * len_ind + sum(results)
    return result,


# -------------------------------------------------------------------------------------- #
def bm_rastrigin_skewed(individual: Individual) -> tuple[float]:
    """
    Skewed Rastrigin test objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness value of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - minimization
          * - Range
            - :math:`x_i \\in [-5.12, 5.12]`
          * - Global optima
            - :math:`x_i = 0, \\forall i \\in \\lbrace 1 \
               \\ldots N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
          * - Function
            - :math:`f(\\mathbf{x}) = 10N + \\sum_{i=1}^N \
               \\left(y_i^2 - 10 \\cos(2\\pi x_i)\\right)`\n
              :math:`\\text{where } y_i = 10\\cdot x_i \
               \\text{ if } x_i > 0 \\text{, else } x_i`
    """
    results = []
    len_ind = len(individual)
    for x in individual:
        var_1 = (10*x if x > 0 else x)**2
        var_2 = 10*cos(2*pi*(10*x if x > 0 else x))
        results.append(var_1 - var_2)
    result = 10 * len_ind + sum(results)
    return result,


# -------------------------------------------------------------------------------------- #
def bm_shekel(individual: Individual, matrix: numpy.ndarray,
              vector: numpy.ndarray) -> tuple[float]:
    """
    The Shekel multimodal function can have any number
    of maxima. The maxima count is given by the length
    of the arguments **matrix** and **vector**.

    :param individual: The Individual to be evaluated.
    :param matrix: Matrix of size :math:`M\\times N`,
        where :math:`M` is the number of maxima and
        :math:`N` is the number of dimensions.
    :param vector: Vector of size :math:`M\\times 1`,
        where :math:`M` is the number of maxima.
    :return: Fitness value of the individual.

    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Type
            - maximization
          * - Range
            - None
          * - Global optima
            - None
          * - Function
            - :math:`f(\\mathbf{x}) = \\sum_{i = 1}^{M} \
               \\frac{1}{c_{i} + \\sum_{j = 1}^{N} \
               (x_{j} - a_{ij})^2 }`
    """
    results = []
    for i in range(len(vector)):
        values = []
        for j, g in enumerate(matrix[i]):
            val = (individual[j] - g) ** 2
            values.append(val)
        result = 1 / (vector[i] + sum(values))
        results.append(result)
    result = sum(results)
    return result,
