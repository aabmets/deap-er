# Persistence

1. `Checkpoint.save` writes a sibling `.tmp` file and replaces the
   destination. A dump that fails part-way through does not truncate
   the last good checkpoint.
2. `Checkpoint.load` restores the constructor's `file_path`,
   `raise_errors`, and `make_dir` after unpickling, so moving a
   checkpoint file does not send the next save back to the old path.
3. Setting `save_freq = -1` while iterating `Checkpoint.range`
   disables further saves. It no longer turns saving on for every
   remaining generation.
