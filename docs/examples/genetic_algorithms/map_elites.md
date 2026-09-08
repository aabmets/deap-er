# MAP-Elites

Illuminates a 2-D behavior grid of Rastrigin with `GridArchive` and
`ea_map_elites`. Fitness is on `ind.fitness`; the descriptor is
the first two genes.

```python
--8<-- "examples/genetic_algorithms/map_elites.py"
```

## CVT archive

`CvtArchive.from_samples` builds Voronoi cells from k-means centroids.
The same `ea_map_elites` loop fills those cells.

```python
--8<-- "examples/genetic_algorithms/map_elites_cvt.py"
```

## Novelty and iso+line

`UnstructuredArchive` keeps elites by descriptor distance.
`sel_novelty` ranks parents by distance to the archive; `mut_iso_line`
interpolates toward a donor elite.

```python
--8<-- "examples/genetic_algorithms/map_elites_novelty.py"
```
