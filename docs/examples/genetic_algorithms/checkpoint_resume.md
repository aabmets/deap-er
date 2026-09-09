# Checkpoint Resume

`Checkpoint.range` persists a short OneMax run; a second constructor
loads the same file and continues. Pass `hof_ind_cls=` so the hall of
fame round-trips as JSON (roadmap
[item 36](../../overview/roadmap/features_31_40.md#36-persistent-hall-of-fame)).
See [Using Checkpoints](../../tutorials/using_checkpoints.md).

```python
--8<-- "examples/genetic_algorithms/checkpoint_resume.py"
```
