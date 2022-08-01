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
from math import exp, sin, cos


__all__ = [
    'bm_ripple', 'bm_sin_cos', 'bm_unwrapped_ball',
    'bm_kotanchek', 'bm_salustowicz_1d', 'bm_salustowicz_2d',
    'bm_rational_polynomial_1', 'bm_rational_polynomial_2'
]


# ====================================================================================== #
def bm_ripple(individual: Individual) -> float:
    """
    Ripple benchmark function.

    :param individual: The individual to be evaluated.
    :return: The fitness of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Range
            - :math:`\\mathbf{x} \\in [-5, 5]^2`
          * - Function
            - :math:`f(\\mathbf{x}) = (x_1 - 3) (x_2 - 3) \
               + 2 \\sin((x_1 - 4) (x_2 -4))`
    """
    i = individual[0]
    j = individual[1]
    a = (i - 3) * (j - 3)
    b = 2 * sin((i - 4) * (j - 4))
    return a + b


# -------------------------------------------------------------------------------------- #
def bm_sin_cos(individual: Individual) -> float:
    """
    Sine cosine benchmark function.

    :param individual: The individual to be evaluated.
    :return: The fitness of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Range
            - :math:`\\mathbf{x} \\in [0, 6]^2`
          * - Function
            - :math:`f(\\mathbf{x}) = 6\\sin(x_1)\\cos(x_2)`
    """
    i = individual[0]
    j = individual[1]
    return 6 * sin(i) * cos(j)


# -------------------------------------------------------------------------------------- #
def bm_unwrapped_ball(individual: Individual) -> float:
    """
    Unwrapped ball benchmark function.

    :param individual: The individual to be evaluated.
    :return: The fitness of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Range
            - :math:`\\mathbf{x} \\in [-2, 8]^n`
          * - Function
            - :math:`f(\\mathbf{x}) = \\frac{10}{5 + \
               \\sum_{i=1}^n (x_i - 3)^2}`
    """
    s = sum((d - 3) ** 2 for d in individual)
    return 10 / (5 + s)


# -------------------------------------------------------------------------------------- #
def bm_kotanchek(individual: Individual) -> float:
    """
    Kotanchek benchmark function.

    :param individual: The individual to be evaluated.
    :return: The fitness of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Range
            - :math:`\\mathbf{x} \\in [-1, 7]^2`
          * - Function
            - :math:`f(\\mathbf{x}) = \\frac{e^{-(x_1 \
                - 1)^2}}{3.2 + (x_2 - 2.5)^2}`
    """
    i = individual[0]
    j = individual[1]
    numer = exp(-(i - 1) ** 2)
    de_nom = 3.2 + (j - 2.5) ** 2
    return numer / de_nom


# -------------------------------------------------------------------------------------- #
def bm_salustowicz_1d(individual: Individual) -> float:
    """
    Salustowicz benchmark function.

    :param individual: The individual to be evaluated.
    :return: The fitness of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Range
            - :math:`x \\in [0, 10]`
          * - Function
            - :math:`f(x) = e^{-x} x^3 \\cos(x) \
               \\sin(x) (\\cos(x) \\sin^2(x) - 1)`
    """
    i = individual[0]
    a = exp(-i) * i ** 3 * cos(i)
    b = sin(i) * (cos(i) * sin(i) ** 2 - 1)
    return a * b


# -------------------------------------------------------------------------------------- #
def bm_salustowicz_2d(individual: Individual) -> float:
    """
    Salustowicz benchmark function.

    :param individual: The individual to be evaluated.
    :return: The fitness of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Range
            - :math:`\\mathbf{x} \\in [0, 7]^2`
          * - Function
            - :math:`f(\\mathbf{x}) = e^{-x_1} x_1^3 \\cos(x_1) \
               \\sin(x_1) (\\cos(x_1) \\sin^2(x_1) - 1) (x_2 -5)`
    """
    i = individual[0]
    j = individual[1]
    a = exp(-i) * i ** 3 * cos(i) * sin(i)
    b = (cos(i) * sin(i) ** 2 - 1) * (j - 5)
    return a * b


# -------------------------------------------------------------------------------------- #
def bm_rational_polynomial_1(individual: Individual) -> float:
    """
    Rational polynomial ball benchmark function.

    :param individual: The individual to be evaluated.
    :return: The fitness of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Range
            - :math:`\\mathbf{x} \\in [0, 2]^3`
          * - Function
            - :math:`f(\\mathbf{x}) = \\frac{30 * (x_1 - 1) \
                (x_3 - 1)}{x_2^2 (x_1 - 10)}`
    """
    i = individual[0]
    j = individual[1]
    k = individual[2]
    numer = 30 * (i - 1) * (k - 1)
    de_nom = (j ** 2 * (i - 10))
    return numer / de_nom


# -------------------------------------------------------------------------------------- #
def bm_rational_polynomial_2(individual: Individual) -> float:
    """
    Rational polynomial benchmark function.

    :param individual: The individual to be evaluated.
    :return: The fitness of the individual.
    :type individual: :ref:`Individual <datatypes>`

    .. dropdown:: Equations
       :margin: 0 5 5 5

       .. list-table::
          :widths: 10 50
          :stub-columns: 1

          * - Range
            - :math:`\\mathbf{x} \\in [0, 6]^2`
          * - Function
            - :math:`f(\\mathbf{x}) = \\frac{(x_1 - 3)^4 + \
                (x_2 - 3)^3 - (x_2 - 3)}{(x_2 - 2)^4 + 10}`
    """
    i = individual[0]
    j = individual[1]
    numer = (i - 3) ** 4 + (j - 3) ** 3 - (j - 3)
    de_nom = ((j - 2) ** 4 + 10)
    return numer / de_nom
