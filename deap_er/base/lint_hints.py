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
from functools import partial
from typing import Callable


# ====================================================================================== #
class LintHints:

    def __test__(self, *_, **__):
        raise NotImplementedError()

    def map(self, *_, **__):
        raise NotImplementedError()

    def clone(self, *_, **__):
        raise NotImplementedError()

    def attr_int(self, *_, **__):
        raise NotImplementedError()

    def attr_bool(self, *_, **__):
        raise NotImplementedError()

    def attr_float(self, *_, **__):
        raise NotImplementedError()

    def attr_item(self, *_, **__):
        raise NotImplementedError()

    def individuals(self, *_, **__):
        raise NotImplementedError()

    def population(self, *_, **__):
        raise NotImplementedError()

    def generate(self, *_, **__):
        raise NotImplementedError()

    def update(self, *_, **__):
        raise NotImplementedError()

    def evaluate(self, *_, **__):
        raise NotImplementedError()

    def select(self, *_, **__):
        raise NotImplementedError()

    def mate(self, *_, **__):
        raise NotImplementedError()

    def mutate(self, *_, **__):
        raise NotImplementedError()
