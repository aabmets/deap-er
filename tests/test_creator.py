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
import subprocess
import sys
import warnings
from typing import Any

import numpy
import pytest
from deap_er import Fitness, creator
from deap_er.private.overrides import ArrayOverride, NumpyOverride

CNAME = "CLASS_NAME"


def test_creator_overwrite_warning():
    old = creator.create_type(CNAME, int)
    with pytest.warns(RuntimeWarning):
        new = creator.create_type(CNAME, float)
    assert new is not old
    assert creator.__dict__[CNAME] is new
    creator.__dict__.pop(CNAME)


def test_create_type_is_idempotent_for_an_equal_definition():
    fit = creator.create_type("IDEM_FIT", Fitness, weights=(1.0,))
    try:
        first = creator.create_type(CNAME, list, fitness=fit, tag="x")
        instance = first([1, 2])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            second = creator.create_type(CNAME, list, fitness=fit, tag="x")
        assert second is first
        assert isinstance(instance, creator.__dict__[CNAME])
    finally:
        creator.__dict__.pop(CNAME)
        creator.__dict__.pop("IDEM_FIT")


def test_create_type_replaces_a_different_definition():
    first = creator.create_type(CNAME, list, tag="x")
    with pytest.warns(RuntimeWarning):
        second: Any = creator.create_type(CNAME, list, tag="y")
    assert second is not first
    assert second.tag == "y"
    creator.__dict__.pop(CNAME)


@pytest.mark.parametrize("name", ["warnings", "array", "cast", "create_type", "NumpyOverride"])
def test_create_type_refuses_module_names(name):
    with pytest.raises(ValueError, match="reserved"):
        creator.create_type(name, list)
    created = creator.create_type(CNAME, array.array("d"))
    assert created([1.5])[0] == 1.5
    creator.__dict__.pop(CNAME)


_PICKLE_SCRIPT = """
import pickle, sys
import dill
from deap_er import Fitness, creator

fit = creator.create_type("PICKLE_FIT", Fitness, weights=(1.0,))
ind_cls = creator.create_type("PICKLE_IND", list, fitness=fit)
if sys.argv[1] == "dump":
    ind = ind_cls([1, 2, 3])
    ind.fitness.values = (4.0,)
    sys.stdout.write(pickle.dumps(ind).hex() + " " + dill.dumps(ind).hex())
else:
    for loads, blob in zip((pickle.loads, dill.loads), sys.stdin.read().split()):
        ind = loads(bytes.fromhex(blob))
        assert type(ind) is creator.PICKLE_IND, type(ind)
        assert ind == [1, 2, 3] and ind.fitness.values == (4.0,)
    print("ok")
"""


def test_created_type_pickles_across_processes():
    def run(mode: str, stdin: str = "") -> str:
        command = [sys.executable, "-c", _PICKLE_SCRIPT, mode]
        result = subprocess.run(command, input=stdin, capture_output=True, text=True, check=True)
        return result.stdout

    assert run("load", run("dump")).strip() == "ok"


class TestCreatorBasicFunctionality:
    def test_creation(self):
        creator.create_type(CNAME, int)
        assert hasattr(creator, CNAME)
        assert issubclass(creator.__dict__[CNAME], int)
        creator.__dict__.pop(CNAME)

    def test_class_attr(self):
        creator.create_type(CNAME, int, my_attr=0)
        assert hasattr(creator.__dict__[CNAME], "my_attr")
        assert hasattr(creator.__dict__[CNAME](), "my_attr")
        creator.__dict__.pop(CNAME)

    def test_instance_attr(self):
        creator.create_type(CNAME, int, my_attr=int)
        assert not hasattr(creator.__dict__[CNAME], "my_attr")
        assert hasattr(creator.__dict__[CNAME](), "my_attr")
        creator.__dict__.pop(CNAME)

    def test_list_creation(self):
        creator.create_type(CNAME, list)
        obj = creator.__dict__[CNAME]([1, 2, 3, 4])
        assert obj == [1, 2, 3, 4]
        creator.__dict__.pop(CNAME)

    def test_list_attr(self):
        creator.create_type(CNAME, list, a=1)
        obj = creator.__dict__[CNAME]([1, 2, 3, 4])
        assert obj.a == 1
        creator.__dict__.pop(CNAME)


class TestCreatorNumpy:
    data = [x for x in range(10)]

    def test_ndarray_class_override(self):
        creator.create_type(CNAME, numpy.ndarray)
        a = creator.__dict__[CNAME]([])
        b = NumpyOverride
        assert isinstance(a, b)
        creator.__dict__.pop(CNAME)

    def test_ndarray_instance_override(self):
        creator.create_type(CNAME, numpy.ndarray([]))
        a = creator.__dict__[CNAME]([])
        b = NumpyOverride
        assert isinstance(a, b)
        creator.__dict__.pop(CNAME)

    def test_ndarray_values(self):
        creator.create_type(CNAME, numpy.ndarray([]))
        obj = creator.__dict__[CNAME](self.data)
        assert all(map(lambda x, y: x == y, obj, self.data))
        creator.__dict__.pop(CNAME)

    def test_array_class_override(self):
        creator.create_type(CNAME, numpy.array)
        a = creator.__dict__[CNAME]([])
        b = NumpyOverride
        assert isinstance(a, b)
        creator.__dict__.pop(CNAME)

    def test_array_instance_override(self):
        creator.create_type(CNAME, numpy.array([]))
        a = creator.__dict__[CNAME]([])
        b = NumpyOverride
        assert isinstance(a, b)
        creator.__dict__.pop(CNAME)

    def test_array_values(self):
        creator.create_type(CNAME, numpy.array([]))
        obj = creator.__dict__[CNAME](self.data)
        assert all(map(lambda x, y: x == y, obj, self.data))
        creator.__dict__.pop(CNAME)

    def test_various_4(self):
        creator.create_type(CNAME, numpy.ndarray)
        a = creator.__dict__[CNAME]([1, 2, 3, 4])
        b = creator.__dict__[CNAME]([5, 6, 7, 8])

        a[1:3], b[1:3] = b[1:3], a[1:3]
        ta = numpy.array([1, 6, 7, 4])
        tb = numpy.array([5, 6, 7, 8])
        assert all(a == ta)
        assert all(b == tb)
        creator.__dict__.pop(CNAME)

    def test_various_5(self):
        creator.create_type(CNAME, numpy.ndarray)
        a = creator.__dict__[CNAME]([1, 2, 3, 4])
        b = creator.__dict__[CNAME]([5, 6, 7, 8])

        a[1:3], b[1:3] = b[1:3].copy(), a[1:3].copy()
        ta = numpy.array([1, 6, 7, 4])
        tb = numpy.array([5, 2, 3, 8])
        assert all(a == ta)
        assert all(b == tb)
        creator.__dict__.pop(CNAME)


class TestCreatorBuiltinsArray:
    data = [x for x in range(10)]

    def test_array_override(self):
        creator.create_type(CNAME, array.array, typecode="i")
        a = creator.__dict__[CNAME]([])
        b = ArrayOverride
        assert isinstance(a, b)
        creator.__dict__.pop(CNAME)

    def test_array_values(self):
        creator.create_type(CNAME, array.array, typecode="i")
        obj = creator.__dict__[CNAME](self.data)
        assert all(map(lambda x, y: x == y, obj, self.data))
        creator.__dict__.pop(CNAME)

    def test_array_comparison(self):
        creator.create_type(CNAME, array.array, typecode="i")
        a = creator.__dict__[CNAME]([1, 2, 3, 4])
        b = creator.__dict__[CNAME]([5, 6, 7, 8])

        a[1:3], b[1:3] = b[1:3], a[1:3]
        ta = array.array("i", [1, 6, 7, 4])
        tb = array.array("i", [5, 2, 3, 8])
        assert a == ta
        assert b == tb
        creator.__dict__.pop(CNAME)

    def test_array_instance_keeps_typecode(self):
        creator.create_type(CNAME, array.array("d"))
        cls = creator.__dict__[CNAME]
        obj = cls([1.5])
        try:
            assert obj.typecode == "d"
            assert obj[0] == 1.5
        finally:
            creator.__dict__.pop(CNAME)
