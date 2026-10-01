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
import pickle
import subprocess
import sys

import pytest
from deap_er import gp
from deap_er.private.programming.primitives import primitive_set_typed

# Each probe runs in a fresh interpreter, where no primitive set has
# registered the ephemeral that the pickled tree holds.
_LOAD_PROBE = """
import pickle
import sys
from deap_er import gp
tree = pickle.loads(sys.stdin.buffer.read())
print(str(tree), type(tree[2]).__name__, tree[2].ret is gp.Window)
pset = gp.columnar_pset(["level"], window=(2, 48), window_name="gp5_window_2_48")
print(type(tree[2]) in pset.terminals[gp.Window])
"""

_LAMBDA_PROBE = """
import pickle
import sys
try:
    pickle.loads(sys.stdin.buffer.read())
except pickle.UnpicklingError as error:
    print(error)
"""


def _run(probe: str, payload: bytes) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", probe], input=payload, capture_output=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr.decode()
    return completed.stdout.decode()


def _ephemeral(pset, slot, name):
    return next(item for item in pset.terminals[slot] if getattr(item, "__name__", "") == name)


def _window_tree() -> gp.PrimitiveTree:
    pset = gp.columnar_pset(["level"], window=(2, 48), window_name="gp5_window_2_48")
    win_cls = _ephemeral(pset, gp.Window, "gp5_window_2_48")
    return gp.PrimitiveTree([pset.mapping["delay"], pset.mapping["level"], win_cls()])


def test_window_ephemeral_tree_loads_in_a_fresh_process_before_registration():
    tree = _window_tree()

    lines = _run(_LOAD_PROBE, pickle.dumps(tree)).splitlines()

    assert lines == [f"{tree} gp5_window_2_48 True", "True"]


def test_lambda_ephemeral_tree_still_pickles_and_names_the_missing_registration():
    pset = gp.PrimitiveSetTyped("main", [float], float)
    pset.add_primitive(max, [float, float], float, name="max")
    pset.add_ephemeral_constant("gp5_lambda", lambda: 1.5, float)
    eph_cls = _ephemeral(pset, float, "gp5_lambda")
    tree = gp.PrimitiveTree([pset.mapping["max"], pset.mapping["ARG0"], eph_cls()])
    payload = pickle.dumps(tree)

    assert str(pickle.loads(payload)) == str(tree)
    assert "gp5_lambda' is not registered" in _run(_LAMBDA_PROBE, payload)


@pytest.mark.parametrize("registered", [True, False])
def test_pickles_naming_the_old_class_location_still_resolve(registered):
    tree = _window_tree()
    name = "gp5_window_2_48" if registered else "gp5_unregistered"

    if registered:
        assert getattr(primitive_set_typed, name) is type(tree[2])
    else:
        with pytest.raises(AttributeError):
            getattr(primitive_set_typed, name)
