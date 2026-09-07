# Multi-Objective ZDT1

ZDT1 with SMS-EMOA, MOEA/D, or AGE-MOEA-II. Change `SELECTOR` at the top
of the script. The run logs wall time through a `logger`, snapshots each
generation's `fronts`, and round-trips the logbook through JSON.

```python
--8<-- "examples/genetic_algorithms/zdt_mo.py"
```
