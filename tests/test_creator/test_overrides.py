#
#   Apache License 2.0
#
#   Copyright (c) 2022, Mattias Aabmets
#
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#
#   SPDX-License-Identifier: Apache-2.0
#
import array
import pickle
from copy import copy, deepcopy

import numpy
from deap_er import base, creator
from deap_er.creator import overrides as ovr


class TestNumpyOverrideClass:
    def test_numpy_override_instantiation(self):
        data = [x for x in range(0, 10)]
        obj = ovr._NumpyOverride(data)
        assert isinstance(obj, ovr._NumpyOverride)
        assert issubclass(ovr._NumpyOverride, numpy.ndarray)

    def test_numpy_override_deepcopy(self):
        data = [x for x in range(0, 10)]
        obj = ovr._NumpyOverride(data)
        copy = deepcopy(obj)
        assert isinstance(copy, numpy.ndarray)
        assert all(obj == copy)
        assert obj.__dict__ == copy.__dict__

    def test_numpy_override_pickling(self):
        data = [x for x in range(0, 10)]
        obj = ovr._NumpyOverride(data)
        jar = pickle.dumps(obj)
        copy = pickle.loads(jar)
        assert all(obj == copy)
        assert obj.__dict__ == copy.__dict__

    def test_array_override_reduction(self):
        data = [x for x in range(0, 10)]
        obj = ovr._NumpyOverride(data)
        cls, args, state = obj.__reduce__()
        assert cls == ovr._NumpyOverride
        assert isinstance(args, tuple)
        assert isinstance(state, dict)


class TestArrayOverrideClass:
    ovr._ArrayOverride.typecode = "i"

    def test_array_override_instantiation(self):
        data = [x for x in range(0, 10)]
        obj = ovr._ArrayOverride(data)
        assert isinstance(obj, ovr._ArrayOverride)
        assert issubclass(ovr._ArrayOverride, array.array)

    def test_array_override_deepcopy(self):
        data = [x for x in range(0, 10)]
        obj = ovr._ArrayOverride(data)
        copy = deepcopy(obj)
        assert isinstance(copy, array.array)
        assert obj == copy
        assert obj.__dict__ == copy.__dict__

    def test_array_override_pickling(self):
        data = [x for x in range(0, 10)]
        obj = ovr._ArrayOverride(data)
        jar = pickle.dumps(obj)
        copy = pickle.loads(jar)
        assert obj == copy
        assert obj.__dict__ == copy.__dict__

    def test_array_override_reduction(self):
        data = [x for x in range(0, 10)]
        obj = ovr._ArrayOverride(data)
        cls, args, state = obj.__reduce__()
        assert cls == ovr._ArrayOverride
        assert isinstance(args, tuple)
        assert isinstance(state, dict)


def test_copy_copy_keeps_type_and_fitness_on_array_individuals():
    fit_name = "OVR_COPY_FIT"
    np_name = "OVR_COPY_NP"
    ar_name = "OVR_COPY_AR"
    creator.create(fit_name, base.Fitness, weights=(1.0,))
    creator.create(np_name, numpy.ndarray, fitness=creator.__dict__[fit_name])
    creator.create(ar_name, array.array, typecode="i", fitness=creator.__dict__[fit_name])
    try:
        np_ind = creator.__dict__[np_name]([1, 2, 3])
        np_ind.fitness.values = (1.5,)
        ar_ind = creator.__dict__[ar_name]([1, 2, 3])
        ar_ind.fitness.values = (2.5,)
        for original in (np_ind, ar_ind):
            shallow = copy(original)
            deep = deepcopy(original)
            assert type(shallow) is type(original)
            assert type(deep) is type(original)
            assert shallow.fitness.values == original.fitness.values
            assert deep.fitness.values == original.fitness.values
            assert hasattr(shallow, "fitness")
    finally:
        del creator.__dict__[np_name]
        del creator.__dict__[ar_name]
        del creator.__dict__[fit_name]
