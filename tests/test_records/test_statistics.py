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
from deap_er.records import Statistics, MultiStatistics
from operator import itemgetter
import numpy


# ====================================================================================== #
class TestStatistics:
    def test_statistics(self):
        s = Statistics()
        s.register("mean", numpy.mean)
        s.register("max", max)
        res = s.compile([1, 2, 3, 4])
        assert res == {'max': 4, 'mean': 2.5}
        res = s.compile([5, 6, 7, 8])
        assert res == {'mean': 6.5, 'max': 8}

    def test_multi_statistics(self):
        length_stats = Statistics(key=len)
        item_stats = Statistics(key=itemgetter(0))
        ms = MultiStatistics(length=length_stats, item=item_stats)
        ms.register("mean", numpy.mean, axis=0)
        ms.register("max", numpy.max, axis=0)
        res = ms.compile([[0.0, 1.0, 1.0, 5.0], [2.0, 5.0]])
        assert res == dict(
            length={'mean': 3.0, 'max': 4},
            item={'mean': 1.0, 'max': 2.0}
        )
