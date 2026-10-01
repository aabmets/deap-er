# Persistence

1. `Checkpoint.save` writes a uniquely named sibling staging file,
   fsyncs it, replaces the destination, and fsyncs the directory. A
   dump that fails part-way through does not truncate the last good
   checkpoint, two writers of one file cannot publish a torn
   checkpoint, and any exception while saving is a save error that
   removes the staging file. State is pickled with the C pickler
   first and with dill only when that fails or references
   `__main__`.
2. `Checkpoint.load` restores the constructor's `file_path`,
   `raise_errors`, and `make_dir` after unpickling, so moving a
   checkpoint file does not send the next save back to the old path.
3. Setting `save_freq = -1` while iterating `Checkpoint.range`
   disables further saves. It no longer turns saving on for every
   remaining generation.
4. `Checkpoint` autoload skips a missing file, so
   `raise_errors=True` no longer fails the first run. Every
   deserialisation failure (a pickled `creator` type this process
   has not created, a truncated file, a payload that is not a
   `dict`, a malformed RNG state) raises `CheckpointError` or
   returns `False`, and a failed load changes neither the instance
   nor the library RNG. `hof` saved without `hof_ind_cls` is
   pickled and survives the round trip instead of disappearing.
