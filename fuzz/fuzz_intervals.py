import atheris

with atheris.instrument_imports():
    import numcodecs as numcodecs
    import numpy as np

import pandas as pandas
import sympy as sympy
import xarray as xarray

with atheris.instrument_imports():
    import sys
    import warnings

    from numcodecs_safeguards.intervals import Interval, IntervalUnion, Lower, Upper


warnings.filterwarnings("error")


def generate_interval_union(
    data: atheris.FuzzedDataProvider, n: int, imin: int, imax: int
) -> tuple[IntervalUnion, set]:
    n = data.ConsumeIntInRange(1, n)

    pivots = sorted(data.ConsumeIntInRange(imin, imax) for _ in range(n * 2))

    intervals = IntervalUnion.empty(np.dtype(int), 1, 1)
    elems = set()

    for i in range(n):
        low, high = pivots[i * 2], pivots[i * 2 + 1]
        interval = Interval.empty(np.dtype(int), 1)
        Lower(np.array(low)) <= interval[:] <= Upper(np.array(high))
        intervals = intervals.union(interval.into_union())
        elems = elems.union(range(low, high + 1))

    return (intervals, elems)


def check_one_input(data):
    data = atheris.FuzzedDataProvider(data)

    n, imin, imax, m = 5, 0, 100, 10

    info = []

    try:
        intervals = IntervalUnion.empty(np.dtype(int), 1, 1)
        elems = set()

        # generate #m interval unions with 1-#n intervals each
        #  and union/intersect them
        for i in range(m):
            ins, els = generate_interval_union(data, n, imin, imax)

            info.append(ins)
            info.append(sorted(els))

            if (i == 0) or data.ConsumeBool():
                info.insert(-2, "|")
                intervals = intervals.union(ins)
                elems = elems.union(els)
            else:
                info.insert(-2, "&")
                intervals = intervals.intersect(ins)
                elems = elems.intersection(els)

            info.append("=")
            info.append(intervals)
            info.append(sorted(elems))

        # compute low/high bounds from the elems to check that there are no
        #  adjacent intervals that should have been merged
        i = None
        lows, highs = [], []
        for e in sorted(elems):
            if i is None:
                i = e
                lows.append(e)
                continue
            if e == i + 1:
                i = e
                continue
            highs.append(i)
            i = e
            lows.append(e)
        if i is not None:
            highs.append(i)

        info.append(lows)
        info.append(highs)

        # one element interval union, so no intervals should be empty
        assert len(lows) == intervals._lower.shape[0]
        assert len(highs) == intervals._lower.shape[0]

        check_elems = set()

        for i in range(intervals._lower.shape[0]):
            low, high = intervals._lower[i, 0], intervals._upper[i, 0]

            assert lows[i] == low
            assert highs[i] == high

            check_elems = check_elems.union(range(low, high + 1))

        assert sorted(elems) == sorted(check_elems)
    except Exception as err:
        print("\n===\n\n" + "\n".join(repr(i) for i in info) + "\n\n===\n")
        raise err


atheris.Setup(sys.argv, check_one_input)
atheris.Fuzz()
