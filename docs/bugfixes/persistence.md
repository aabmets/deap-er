# Persistence

Correctness fixes in `Checkpoint` save, load, and `range`.

---

## `save` truncates the good file before dumping

`open(path, "wb")` wiped the previous checkpoint, then `dill.dump`
ran. A pickle/`TypeError` mid-write left a truncated file.
`raise_errors=False` still raised `TypeError` / `EOFError` because
those are not `dill.PickleError`.

**Fix.** Dump to a sibling `*.tmp` and `os.replace` onto `file_path`
only after a successful dump. Load and save also catch `EOFError` and
`TypeError`. A failed dump leaves the previous file loadable.

**Validator.**
`tests/test_controllers/test_checkpoint.py::TestCheckpoint::test_failed_save_leaves_previous_checkpoint_loadable`

---

## `load` overwrites constructor `file_path`, `raise_errors`, and `make_dir`

`load` rebound `self.__dict__` from the pickle, including the original
absolute path and error-handling flags. Opening a copied checkpoint
with a new `dir_path` still wrote later saves to the old location.

**Fix.** Stash `file_path`, `raise_errors`, and `make_dir` before the
rebind, then restore those constructor fields (including on error).

**Validator.**
`tests/test_controllers/test_checkpoint.py::TestCheckpoint::test_load_keeps_constructor_path_and_raise_errors`

---

## Mid-loop `save_freq = -1` turns saving on

The enabled/disabled switch was tested once before the first yield.
Setting `save_freq = -1` inside the enabled loop made
`elapsed >= -1.0` always true, so every remaining generation saved.
The reverse switch (enable after starting disabled) did nothing.

**Fix.** `range` decides whether to save **inside** the loop from the
current `save_freq` (`-1` means skip).

**Validator.**
`tests/test_controllers/test_checkpoint.py::TestCheckpoint::test_save_freq_can_be_disabled_during_range`
