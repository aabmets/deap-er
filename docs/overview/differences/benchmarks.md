# Benchmarks

1. DTLZ5 / DTLZ6 apply the angular $\theta$ map only to the first
   $M-1$ decision variables. Distance variables feed $g$ and are not
   multiplied into $f_1$ as extra cosines.
2. Chuang F3 scores the selector-1 branch with the published trap
   (not the inverse trap) and covers bits $0..39$ without dropping
   bit 38.
3. Moving Peaks `ALT1` uses `move_severity` $1.5$, matching its
   documented table.
4. Royal Road R2 sums R1 at every doubling of `order` that still
   fits the bit string. The top-level schema is included, so a
   64-bit all-ones individual with order $8$ scores $256$.
5. Kotanchek uses the published denominator $1.2`, not $3.2$.
6. Royal Road R1 (and R2, which sums R1) and `bin2float` treat
   boolean bits like integers. `mut_flip_bit` preserves `bool`;
   bits no longer stringify as `"True"` / `"False"` and raise
   `ValueError`.
