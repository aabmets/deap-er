import numpy
from deap_er import gp, tools

tools.rng.seed(1234)  # disables randomization

COLUMNS = ["level", "flow"]
ROWS = 24


def make_book():
    steps = numpy.linspace(0.0, 6.0, ROWS)
    level = numpy.sin(steps) + 0.1 * steps
    flow = numpy.cos(steps * 0.8)
    columns = tuple(numpy.ascontiguousarray(c, dtype=numpy.float64) for c in (level, flow))
    matrix = numpy.stack(columns, axis=1)
    target = gp.rolling_mean(level, 4) - gp.delay(flow, 2)
    return columns, matrix, target


def certificate(tape, bounds, n_rows):
    flags = gp.tape_flags(tape, bounds, n_rows=n_rows)
    lo, hi = gp.tape_interval(tape, bounds)
    return flags, lo, hi


def main():
    _columns, matrix, target = make_book()
    bounds = gp.bounds_from_matrix(matrix)
    pset = gp.columnar_pset(COLUMNS, window=(2, 32))

    good_tree = gp.PrimitiveTree.from_string("rolling_mean(level, 4)", pset)
    bad_tree = gp.PrimitiveTree.from_string("rolling_mean(level, 32)", pset)
    good_tape = gp.lower_tree(good_tree, pset)
    bad_tape = gp.lower_tree(bad_tree, pset)

    good_flags, good_lo, good_hi = certificate(good_tape, bounds, ROWS)
    bad_flags, _bad_lo, _bad_hi = certificate(bad_tape, bounds, ROWS)

    if good_flags.all_nan or not bad_flags.all_nan:
        raise RuntimeError("Interval certificates did not separate viable and dead tapes.")

    individuals = [good_tree, bad_tree]
    scores = gp.evaluate_columnar(
        individuals,
        pset,
        matrix,
        target,
        backend="opcode",
        static_filter=True,
        empty=1.0e6,
    )
    if scores[0][0] >= 1.0e5 or scores[1][0] != 1.0e6:
        raise RuntimeError("Static filter did not skip the dead-window program.")
    print(f"\nGood program MSE: {scores[0][0]:.4g} (interval [{good_lo:.3g}, {good_hi:.3g}])")
    print(f"Dead-window program sentinel: {scores[1][0]:.4g} (all_nan={bad_flags.all_nan})")


if __name__ == "__main__":
    main()
