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

## Namespaces

The functionality of this library is divided into the following namespaces:

- **deap_er** — `Toolbox`, `Fitness`, `Checkpoint`, and the `creator` module.
- **tools** — Essential evolution components:
    - algorithms
    - operators
    - strategies
    - records
    - utilities
    - benchmarks
- **gp** — Components for genetic programming.

These namespaces can be imported with:

```python
from deap_er import Checkpoint, Fitness, Toolbox, creator, gp, tools
```
