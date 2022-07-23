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
from math import sin, cos, pi, exp, sqrt
from functools import reduce
from operator import mul


__all__ = [
    'kursawe', 'schaffer_mo', 'fonseca', 'poloni', 'dent',
    'zdt_1', 'zdt_2', 'zdt_3', 'zdt_4', 'zdt_6',
    'dtlz_1', 'dtlz_2', 'dtlz_3', 'dtlz_4', 'dtlz_5', 'dtlz_6', 'dtlz_7'
]


# ====================================================================================== #
def kursawe(individual: Individual) -> tuple[float, float]:
    """
    Kursawe multi-objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`f_{1}(\\mathbf{x}) = \\sum_{i=1}^{N-1} -10 e^{-0.2 \\sqrt{x_i^2 + x_{i+1}^2} }`\n
       :math:`f_{2}(\\mathbf{x}) = \\sum_{i=1}^{N} |x_i|^{0.8} + 5 \\sin(x_i^3)`\n
       Returns :math:`f_{1}(\\mathbf{x})` and :math:`f_{2}(\\mathbf{x})`.
    """
    def fn(x, y):
        return -10 * exp(-0.2 * sqrt(x * x + y * y))

    f1 = sum(fn(x, y) for x, y in zip(individual[:-1], individual[1:]))
    f2 = sum(abs(x) ** 0.8 + 5 * sin(x * x * x) for x in individual)
    return f1, f2


# -------------------------------------------------------------------------------------- #
def schaffer_mo(individual: Individual) -> tuple[float, float]:
    """
    Schaffer's multi-objective function on a one-attribute **individual**.

    :param individual: The Individual to be evaluated.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`f_{1}(\\mathbf{x}) = x_1^2`\n
       :math:`f_{2}(\\mathbf{x}) = (x_1-2)^2`\n
       Returns :math:`f_{1}(\\mathbf{x})` and :math:`f_{2}(\\mathbf{x})`.
    """
    f1 = individual[0] ** 2
    f2 = (individual[0] - 2) ** 2
    return f1, f2


# -------------------------------------------------------------------------------------- #
def fonseca(individual: Individual) -> tuple[float, float]:
    """
    Fonseca and Fleming's multiobjective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`f_{1}(\\mathbf{x}) = 1 - e^{-\\sum_{i=1}^{3}(x_i - \\frac{1}{\\sqrt{3}})^2}`\n
       :math:`f_{2}(\\mathbf{x}) = 1 - e^{-\\sum_{i=1}^{3}(x_i + \\frac{1}{\\sqrt{3}})^2}`\n
       Returns :math:`f_{1}(\\mathbf{x})` and :math:`f_{2}(\\mathbf{x})`.
    """
    f1 = 1 - exp(-sum((xi - 1/sqrt(3))**2 for xi in individual[:3]))
    f2 = 1 - exp(-sum((xi + 1/sqrt(3))**2 for xi in individual[:3]))
    return f1, f2


# -------------------------------------------------------------------------------------- #
def poloni(individual: Individual) -> tuple[float, float]:
    """
    Poloni's multiobjective function on a two-attribute **individual**.

    :param individual: The Individual to be evaluated.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`A_1 = 0.5 \\sin (1) - 2 \\cos (1) + \\sin (2) - 1.5 \\cos (2)`\n
       :math:`A_2 = 1.5 \\sin (1) - \\cos (1) + 2 \\sin (2) - 0.5 \\cos (2)`
       :math:`B_1 = 0.5 \\sin (x_1) - 2 \\cos (x_1) + \\sin (x_2) - 1.5 \\cos (x_2)`\n
       :math:`B_2 = 1.5 \\sin (x_1) - cos(x_1) + 2 \\sin (x_2) - 0.5 \\cos (x_2)`
       :math:`f_{1}(\\mathbf{x}) = 1 + (A_1 - B_1)^2 + (A_2 - B_2)^2`\n
       :math:`f_{2}(\\mathbf{x}) = (x_1 + 3)^2 + (x_2 + 1)^2`
       Returns :math:`f_{1}(\\mathbf{x})` and :math:`f_{2}(\\mathbf{x})`.
    """
    x_1 = individual[0]
    x_2 = individual[1]
    a_1 = 0.5 * sin(1) - 2 * cos(1) + sin(2) - 1.5 * cos(2)
    a_2 = 1.5 * sin(1) - cos(1) + 2 * sin(2) - 0.5 * cos(2)
    b_1 = 0.5 * sin(x_1) - 2 * cos(x_1) + sin(x_2) - 1.5 * cos(x_2)
    b_2 = 1.5 * sin(x_1) - cos(x_1) + 2 * sin(x_2) - 0.5 * cos(x_2)
    f1 = 1 + (a_1 - b_1) ** 2 + (a_2 - b_2) ** 2
    f2 = (x_1 + 3) ** 2 + (x_2 + 1) ** 2
    return f1, f2


# -------------------------------------------------------------------------------------- #
def dent(individual: Individual, dent_size: float = 0.85) -> tuple[float, float]:
    """
    | Two-objective problem with a "dent". The **individual** must have
    | two attributes that take values in the range of [-1.5, 1.5].

    :param individual: The Individual to be evaluated.
    :param dent_size: The size of the dent.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`f_{1}(\\mathbf{x}) = ?`\n
       :math:`f_{2}(\\mathbf{x}) = ?`\n
       Returns :math:`f_{1}(\\mathbf{x})` and :math:`f_{2}(\\mathbf{x})`.
    """
    d = dent_size * exp(-(individual[0] - individual[1]) ** 2)
    f1 = 0.5 * (sqrt(1 + (individual[0] + individual[1]) ** 2) +
                sqrt(1 + (individual[0] - individual[1]) ** 2) +
                individual[0] - individual[1]) + d
    f2 = 0.5 * (sqrt(1 + (individual[0] + individual[1]) ** 2) +
                sqrt(1 + (individual[0] - individual[1]) ** 2) -
                individual[0] + individual[1]) + d
    return f1, f2


# -------------------------------------------------------------------------------------- #
def zdt_1(individual: Individual) -> tuple[float, float]:
    """
    ZDT1 multi-objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\\mathbf{x}) = 1 + \\frac{9}{n-1}\\sum_{i=2}^n x_i`\n
       :math:`f_{1}(\\mathbf{x}) = x_1`\n
       :math:`f_{2}(\\mathbf{x}) = g(\\mathbf{x})\\left[1 - \
            \\sqrt{\\frac{x_1}{g(\\mathbf{x})}}\\right]`\n
       Returns :math:`f_{1}(\\mathbf{x})` and :math:`f_{2}(\\mathbf{x})`.
    """
    g = 1.0 + 9.0 * sum(individual[1:]) / (len(individual) - 1)
    f1 = individual[0]
    f2 = g * (1 - sqrt(f1 / g))
    return f1, f2


# -------------------------------------------------------------------------------------- #
def zdt_2(individual: Individual) -> tuple[float, float]:
    """
    ZDT2 multi-objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\\mathbf{x}) = 1 + \\frac{9}{n-1}\\sum_{i=2}^n x_i`\n
       :math:`f_{1}(\\mathbf{x}) = x_1`\n
       :math:`f_{2}(\\mathbf{x}) = g(\\mathbf{x})\\left[1 - \
            \\left(\\frac{x_1}{g(\\mathbf{x})}\\right)^2\\right]`\n
       Returns :math:`f_{1}(\\mathbf{x})` and :math:`f_{2}(\\mathbf{x})`.
    """
    g = 1.0 + 9.0*sum(individual[1:])/(len(individual)-1)
    f1 = individual[0]
    f2 = g * (1 - (f1 / g) ** 2)
    return f1, f2


# -------------------------------------------------------------------------------------- #
def zdt_3(individual: Individual) -> tuple[float, float]:
    """
    ZDT3 multi-objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\\mathbf{x}) = 1 + \\frac{9}{n-1}\\sum_{i=2}^n x_i`\n
       :math:`f_{1}(\\mathbf{x}) = x_1`\n
       :math:`f_{2}(\\mathbf{x}) = g(\\mathbf{x})\\left[1 - \
            \\sqrt{\\frac{x_1}{g(\\mathbf{x})}} - \\frac{x_1}{g(\\mathbf{x})} \
            \\sin(10\\pi x_1)\\right]`\n
       Returns :math:`f_{1}(\\mathbf{x})` and :math:`f_{2}(\\mathbf{x})`.
    """
    g = 1.0 + 9.0*sum(individual[1:])/(len(individual)-1)
    f1 = individual[0]
    f2 = g * (1 - sqrt(f1 / g) - f1 / g * sin(10 * pi * f1))
    return f1, f2


# -------------------------------------------------------------------------------------- #
def zdt_4(individual: Individual) -> tuple[float, float]:
    """
    ZDT4 multi-objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\\mathbf{x}) = 1 + 10(n-1) + \\sum_{i=2}^n \
            \\left[ x_i^2 - 10\\cos(4\\pi x_i) \\right]`\n
       :math:`f_{1}(\\mathbf{x}) = x_1`\n
       :math:`f_{2}(\\mathbf{x}) = g(\\mathbf{x}) \\left[ 1 - \
            \\sqrt{ \\frac{x_1}{g(\\mathbf{x})}} \\right]`\n
       Returns :math:`f_{1}(\\mathbf{x})` and :math:`f_{2}(\\mathbf{x})`.
    """
    var = sum(xi ** 2 - 10 * cos(4 * pi * xi) for xi in individual[1:])
    g = 1 + 10 * (len(individual)-1) + var
    f1 = individual[0]
    f2 = g * (1 - sqrt(f1 / g))
    return f1, f2


# -------------------------------------------------------------------------------------- #
def zdt_6(individual: Individual) -> tuple[float, float]:
    """
    ZDT6 multi-objective function.

    :param individual: The Individual to be evaluated.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\\mathbf{x}) = 1 + 9 \\left[ \\left(\\sum_{i=2}^n \
            x_i\\right)/(n-1) \\right]^{0.25}`\n
       :math:`f_{1}(\\mathbf{x}) = 1 - \\exp(-4x_1)\\sin^6(6\\pi x_1)`\n
       :math:`f_{2}(\\mathbf{x}) = g(\\mathbf{x}) \\left[1 - \\left( \
            \\frac{f_{1}(\\mathbf{x})}{g(\\mathbf{x})}\\right)^2 \\right]`\n
       Returns :math:`f_{1}(\\mathbf{x})` and :math:`f_{2}(\\mathbf{x})`.
    """
    g = 1 + 9 * (sum(individual[1:]) / (len(individual)-1)) ** 0.25
    f1 = 1 - exp(-4 * individual[0]) * sin(6 * pi * individual[0]) ** 6
    f2 = g * (1 - (f1 / g) ** 2)
    return f1, f2


# -------------------------------------------------------------------------------------- #
def dtlz_1(individual: Individual, count: int) -> list:
    """
    | DTLZ1 multi-objective function. Returns a list of size **count**.
    | The **individual** must have at least **count** number of elements.

    :param individual: The Individual to be evaluated.
    :param count: Number of objectives.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\\mathbf{x}_m) = 100\\left(|\\mathbf{x}_m| + \\sum_{x_i \
            \\in \\mathbf{x}_m}\\left((x_i - 0.5)^2 - \
            \\cos(20\\pi(x_i - 0.5))\\right)\\right)`\n
       :math:`f_{1}(\\mathbf{x}) = \\frac{1}{2} (1 + \
            g(\\mathbf{x}_m)) \\prod_{i=1}^{m-1}x_i`\n
       :math:`f_{2}(\\mathbf{x}) = \\frac{1}{2} (1 + g(\\mathbf{x}_m)) \
            (1-x_{m-1}) \\prod_{i=1}^{m-2}x_i`\n
       :math:`f_{m-1}(\\mathbf{x}) = \\frac{1}{2} (1 + \
            g(\\mathbf{x}_m)) (1 - x_2) x_1`\n
       :math:`\\ldots`\n
       :math:`f_{m}(\\mathbf{x}) = \\frac{1}{2} \
            (1 - x_1)(1 + g(\\mathbf{x}_m))`\n

       Where :math:`m` is the number of objectives and :math:`\\mathbf{x}_m`
       is a vector of the remaining attributes :math:`[x_m~\\ldots~x_n]`
       of the individual in :math:`n > m` dimensions.
    """
    def fn_xi(xi):
        _cos = cos(20 * pi * (xi - 0.5))
        return (xi - 0.5) ** 2 - _cos

    def fn_m(m):
        rdc = reduce(mul, individual[:m], 1)
        return 0.5 * rdc * (1 - individual[m]) * (1 + gval)

    _sum = sum(fn_xi(xi) for xi in individual[count-1:])
    gval = 100 * (len(individual[count-1:]) + _sum)
    fit = [0.5 * reduce(mul, individual[:count-1], 1) * (1 + gval)]
    fit.extend(fn_m(m) for m in reversed(range(count-1)))
    return fit


# -------------------------------------------------------------------------------------- #
def dtlz_2(individual: Individual, count: int) -> list:
    """
    | DTLZ2 multi-objective function. Returns a list of size **count**.
    | The **individual** must have at least **count** number of elements.

    :param individual: The Individual to be evaluated.
    :param count: Number of objectives.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\\mathbf{x}_m) = \\sum_{x_i \\in \
            \\mathbf{x}_m} (x_i - 0.5)^2`\n
       :math:`f_{1}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \
            \\prod_{i=1}^{m-1} \\cos(0.5x_i\\pi)`\n
       :math:`f_{2}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \
            \\sin(0.5x_{m-1}\\pi ) \\prod_{i=1}^{m-2} \\cos(0.5x_i\\pi)`\n
       :math:`\\ldots`\n
       :math:`f_{m}(\\mathbf{x}) = (1 + \
            g(\\mathbf{x}_m)) \\sin(0.5x_{1}\\pi )`\n

       Where :math:`m` is the number of objectives and :math:`\\mathbf{x}_m`
       is a vector of the remaining attributes :math:`[x_m~\\ldots~x_n]`
       of the individual in :math:`n > m` dimensions.
    """
    xm = individual[count - 1:]
    gval = sum((xi - 0.5) ** 2 for xi in xm)
    return _dtlz_helper_1(individual, count, gval)


# -------------------------------------------------------------------------------------- #
def dtlz_3(individual: Individual, count: int) -> list:
    """
    | DTLZ3 multi-objective function. Returns a list of size **count**.
    | The **individual** must have at least **count** number of elements.

    :param individual: The Individual to be evaluated.
    :param count: Number of objectives.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\\mathbf{x}_m) = 100\\left(|\\mathbf{x}_m| + \
            \\sum_{x_i \\in \\mathbf{x}_m}\\left((x_i - 0.5)^2 - \
            \\cos(20\\pi(x_i - 0.5))\\right)\\right)`\n
       :math:`f_{1}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \
            \\prod_{i=1}^{m-1} \\cos(0.5x_i\\pi)`\n
       :math:`f_{2}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \
            \\sin(0.5x_{m-1}\\pi ) \\prod_{i=1}^{m-2} \\cos(0.5x_i\\pi)`\n
       :math:`\\ldots`\n
       :math:`f_{m}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \
            \\sin(0.5x_{1}\\pi )`\n

       Where :math:`m` is the number of objectives and :math:`\\mathbf{x}_m`
       is a vector of the remaining attributes :math:`[x_m~\\ldots~x_n]`
       of the individual in :math:`n > m` dimensions.
    """
    def fn(xi):
        _cos = cos(20 * pi * (xi - 0.5))
        return (xi - 0.5) ** 2 - _cos

    xm = individual[count - 1:]
    gval = 100 * (len(xm) + sum(fn(xi) for xi in xm))
    return _dtlz_helper_1(individual, count, gval)


# -------------------------------------------------------------------------------------- #
def dtlz_4(individual: Individual, count: int, alpha: float) -> list:
    """
    | DTLZ4 multi-objective function. Returns a list of size **count**.
    | The **individual** must have at least **count** number of elements.

    :param individual: The Individual to be evaluated.
    :param count: Number of objectives.
    :param alpha: Fitness values exponentiation factor.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\\mathbf{x}_m) = \\sum_{x_i \\in \
            \\mathbf{x}_m} (x_i - 0.5)^2`\n
       :math:`f_{1}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \
            \\prod_{i=1}^{m-1} \\cos(0.5x_i^\\alpha\\pi)`\n
       :math:`f_{2}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \
            \\sin(0.5x_{m-1}^\\alpha\\pi ) \\prod_{i=1}^{m-2} \
            \\cos(0.5x_i^\\alpha\\pi)`\n
       :math:`\\ldots`\n
       :math:`f_{m}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \
            \\sin(0.5x_{1}^\\alpha\\pi )`\n

       Where :math:`m` is the number of objectives and :math:`\\mathbf{x}_m`
       is a vector of the remaining attributes :math:`[x_m~\\ldots~x_n]`
       of the individual in :math:`n > m` dimensions.
    """
    xm = individual[count - 1:]
    gval = sum((xi - 0.5) ** 2 for xi in xm)
    return _dtlz_helper_1(individual, count, gval, alpha)


# -------------------------------------------------------------------------------------- #
def dtlz_5(individual: Individual, count: int) -> list:
    """
    | DTLZ5 multi-objective function. Returns a list of size **count**.
    | The **individual** must have at least **count** number of elements.

    :param individual: The Individual to be evaluated.
    :param count: Number of objectives.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\\mathbf{x}_m) = \\text{ ?}`\n
       :math:`f_{1}(\\mathbf{x}) = \\text{ ?}`\n
       :math:`f_{2}(\\mathbf{x}) = \\text{ ?}`\n
       :math:`\\ldots`\n
       :math:`f_{m}(\\mathbf{x}) = \\text{ ?}`\n

       Where :math:`m` is the number of objectives and :math:`\\mathbf{x}_m`
       is a vector of the remaining attributes :math:`[x_m~\\ldots~x_n]`
       of the individual in :math:`n > m` dimensions.
    """
    gval = sum([(a - 0.5) ** 2 for a in individual[count - 1:]])
    return _dtlz_helper_2(individual, count, gval)


# -------------------------------------------------------------------------------------- #
def dtlz_6(individual: Individual, count: int) -> list:
    """
    | DTLZ6 multi-objective function. Returns a list of size **count**.
    | The **individual** must have at least **count** number of elements.

    :param individual: The Individual to be evaluated.
    :param count: Number of objectives.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\\mathbf{x}_m) = \\text{ ?}`\n
       :math:`f_{1}(\\mathbf{x}) = \\text{ ?}`\n
       :math:`f_{2}(\\mathbf{x}) = \\text{ ?}`\n
       :math:`\\ldots`\n
       :math:`f_{m}(\\mathbf{x}) = \\text{ ?}`\n

       Where :math:`m` is the number of objectives and :math:`\\mathbf{x}_m`
       is a vector of the remaining attributes :math:`[x_m~\\ldots~x_n]`
       of the individual in :math:`n > m` dimensions.
    """
    gval = sum([a ** 0.1 for a in individual[count - 1:]])
    return _dtlz_helper_2(individual, count, gval)


# -------------------------------------------------------------------------------------- #
def dtlz_7(individual: Individual, count: int) -> list:
    """
    | DTLZ7 multi-objective function. Returns a list of size **count**.
    | The **individual** must have at least **count** number of elements.

    :param individual: The Individual to be evaluated.
    :param count: Number of objectives.
    :return: Fitness values of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       :math:`g(\\mathbf{x}_m) = \\text{ ?}`\n
       :math:`f_{1}(\\mathbf{x}) = \\text{ ?}`\n
       :math:`f_{2}(\\mathbf{x}) = \\text{ ?}`\n
       :math:`\\ldots`\n
       :math:`f_{m}(\\mathbf{x}) = \\text{ ?}`\n

       Where :math:`m` is the number of objectives and :math:`\\mathbf{x}_m`
       is a vector of the remaining attributes :math:`[x_m~\\ldots~x_n]`
       of the individual in :math:`n > m` dimensions.
    """
    def fn(a):
        return a / (1 + gval) * (1 + sin(3 * pi * a))

    gval = sum([a for a in individual[count - 1:]])
    gval = 1 + 9 / len(individual[count - 1:]) * gval

    fit = [x for x in individual[:count - 1]]
    vals = [fn(a) for a in individual[:count - 1]]
    res = (1 + gval) * (count - sum(vals))
    fit.append(res)
    return fit


# -------------------------------------------------------------------------------------- #
def _dtlz_helper_1(individual, count, gval, alpha=1.0) -> list:
    def fn(m):
        vals_ = [cos(0.5 * xi ** alpha * pi) for xi in xc[:m]]
        rdc = reduce(mul, vals_, 1)
        _sin = sin(0.5 * xc[m] ** alpha * pi)
        return (1 + gval) * rdc * _sin

    xc = individual[:count - 1]
    vals = (cos(0.5 * xi ** alpha * pi) for xi in xc)
    fit = [(1 + gval) * reduce(mul, vals, 1)]
    vals = [fn(m) for m in range(count - 2, -1, -1)]
    fit.extend(vals)
    return fit


# -------------------------------------------------------------------------------------- #
def _dtlz_helper_2(individual, count, gval) -> list:
    def theta(x):
        return pi / (4.0 * (1 + gval)) * (1 + 2 * gval * x)

    vals = [cos(theta(a)) for a in individual[1:]]
    rdc = reduce(lambda x, y: x * y, vals)
    fit = [(1 + gval) * cos(pi / 2.0 * individual[0]) * rdc]

    for m in reversed(range(1, count)):
        if m == 1:
            res = (1 + gval) * sin(pi / 2 * individual[0])
        else:
            vals = [cos(theta(a)) for a in individual[1:m - 1]]
            rdc = reduce(lambda x, y: x * y, vals, 1)
            _cos = cos(pi / 2 * individual[0])
            _sin = sin(theta(individual[m - 1]))
            res = (1 + gval) * _cos * rdc * _sin
        fit.append(res)
    return fit
