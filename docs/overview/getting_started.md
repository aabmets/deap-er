# Getting Started

## Installation

This library can be installed with:

```text
pip install deap-er
```

or if you're using the [uv](https://docs.astral.sh/uv/) package manager:

```text
uv add deap-er
```

The optional Numba compile backend for columnar GP is an extra:

```text
pip install deap-er[numba]
```

```text
uv add deap-er --extra numba
```

## Namespaces

The functionality of this library is divided into the following namespaces:

- **deap_er** — `Toolbox`, `Fitness`, `Checkpoint`, and the `creator` module.
- **tools** — algorithms (`ea_*`, MAP-Elites, islands), operators
  (including lexicase, SMS-EMOA, MOEA/D, AGE-MOEA-II, mixed-gene
  variation), CMA strategies (boxed, separable, restarting, MO-CMA),
  records (logbook, hall of fame, MAP-Elites archives), utilities, and
  benchmarks.
- **gp** — prefix-tree GP (loosely typed, strongly typed, ADFs),
  SlimGP, and columnar kits with opcode / Numba backends.

These namespaces can be imported with:

```python
from deap_er import Checkpoint, Fitness, Toolbox, creator, gp, tools
```
