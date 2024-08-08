#
#   MIT License
#   
#   Copyright (c) 2022, Mattias Aabmets
#   
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#   
#   SPDX-License-Identifier: MIT
#
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
