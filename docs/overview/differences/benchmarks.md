# Benchmarks

1. DTLZ5 / DTLZ6 apply the angular $\theta$ map only to the first
   $M-1$ decision variables. Distance variables feed $g$ and are not
   multiplied into $f_1$ as extra cosines.
2. Chuang F3 scores the selector-1 branch with the published trap
   (not the inverse trap) and covers bits $0..39$ without dropping
   bit 38. The wrap block is concatenated, not added elementwise, so
   a NumPy individual scores the same as a list.
3. Moving Peaks `ALT1` uses `move_severity` $1.5$, matching its
   documented table. A `pfunc` pool shorter than `npeaks` fills
   every peak from the pool instead of raising `ValueError`, and a
   negative `uniform_height` or `uniform_width` draws random values,
   as documented for `<= 0`, instead of being used as is.
4. Royal Road R2 sums R1 at every doubling of `order` that still
   fits the bit string. The top-level schema is included, so a
   64-bit all-ones individual with order $8$ scores $256$.
5. Kotanchek uses the published denominator $1.2`, not $3.2$.
6. Royal Road R1 (and R2, which sums R1) and `bin2float` treat
   boolean bits like integers. `mut_flip_bit` preserves `bool`;
   bits no longer stringify as `"True"` / `"False"` and raise
   `ValueError`.
7. Scaled Rastrigin accepts a one-dimensional individual. The
   $10^{(i-1)/(N-1)}$ scale is $1$ at $N = 1$ instead of raising
   `ZeroDivisionError`, so it matches plain Rastrigin.
