from collections.abc import Mapping
from typing import overload

import numpy as np
from typing_extensions import (
    Unpack,  # MSPV 3.11
    override,  # MSPV 3.12
)

from ....utils.bindings import Parameter
from ....utils.typing import S
from ..bound import checked_data_bounds
from ..typing import F, Fi, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant

# how to recreate the monotonicity safeguard:
# strict:
#  strictly increasing/decreasing -> strictly increasing/decreasing
#  eb = 0
#  qoi = where(
#      monotonicity(C["$X"], strict=1),  # check if originally all-const
#      monotonicity(X, strict=1),  # if not, preserve monotonicity
#      NaN,  # if all-const, allow anything
#  )
# strict_with_consts:
#  strictly increasing/const/decreasing -> strictly increasing/const/decreasing
#  eb = 0
#  qoi = monotonicity(X, strict=1)
# strict_to_weak:
#  strictly increasing/decreasing -> weakly increasing/const/decreasing
#  eb = 1
#  qoi = where(
#      monotonicity(C["$X"], strict=1),  # check if originally all-const
#      # if not, preserve monotonicity weakly
#      #  +1 -> [0, +1] (since eb=1)
#      #   0 -> not possible (since where)
#      #  -1 -> [-1, 0] (since eb=1)
#      monotonicity(C["$X"], strict=1) - monotonicity(X, strict=0),
#      NaN,  # if all-const, allow anything
#  )
# weak:
#  weakly increasing/decreasing -> weakly increasing/const/decreasing
#  eb = 1
#  qoi = where(
#      monotonicity(C["$X"], strict=0),  # check if originally all-const
#      # if not, preserve monotonicity weakly
#      #  +1 -> [0, +1] (since eb=1)
#      #   0 -> not possible (since where)
#      #  -1 -> [-1, 0] (since eb=1)
#      monotonicity(C["$X"], strict=0) - monotonicity(X, strict=0),
#      NaN,  # if all-const, allow anything
#  )


class ScalarMonotonicity(Expr[AnyExpr, AnyExpr, Unpack[tuple[AnyExpr, ...]]]):
    __slots__: tuple[str, ...] = ("_a", "_b", "_cs", "_strict")
    _a: AnyExpr
    _b: AnyExpr
    _cs: tuple[AnyExpr, ...]
    _strict: bool

    def __init__(self, a: AnyExpr, b: AnyExpr, *cs: AnyExpr, strict: bool):
        self._a = a
        self._b = b
        self._cs = cs
        self._strict = strict

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr, Unpack[tuple[AnyExpr, ...]]]:
        return (self._a, self._b, *self._cs)

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr, *cs: AnyExpr) -> "ScalarMonotonicity":
        return ScalarMonotonicity(a, b, *cs, strict=self._strict)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        # we first individually fold each term
        fa = self._a.constant_fold(dtype)
        fb = self._b.constant_fold(dtype)
        fcs = [c.constant_fold(dtype) for c in self._cs]

        # we can only constant fold if all terms are constant
        if (
            isinstance(fa, Expr)
            or isinstance(fb, Expr)
            or any(isinstance(fc, Expr) for fc in fcs)
        ):
            return ScalarMonotonicity(
                fa if isinstance(fa, Expr) else ScalarFoldedConstant(fa),
                fb if isinstance(fb, Expr) else ScalarFoldedConstant(fb),
                *tuple(
                    fc if isinstance(fc, Expr) else ScalarFoldedConstant(fc)
                    for fc in fcs
                ),
                strict=self._strict,
            )

        acc: F = _compare(fa, fb)
        prev: F = fa
        next: F = fb
        rest: list[F] = list(fcs)  # type: ignore

        while len(rest) > 0:
            prev = next
            next = rest.pop(0)

            cmp: F = _compare(prev, next)

            if self._strict:
                # strict: all comparisons must have the same result (</=/>)
                if cmp != acc:
                    return dtype.type(np.nan)
            else:
                # weak
                if acc == 0:
                    # (a) move from constant to weakly monotonic
                    acc = cmp
                elif cmp == 0:
                    # (b) keep constant or weakly monotonic as-is
                    pass
                elif cmp != acc:
                    # (c) comparison must not change sign (< vs >)
                    return dtype.type(np.nan)

        return acc

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        # we first individually fold each term
        fa = self._a.eval(Xs, late_bound)
        fb = self._b.eval(Xs, late_bound)
        fcs = [c.eval(Xs, late_bound) for c in self._cs]

        acc: np.ndarray[tuple[Ps], np.dtype[F]] = _compare(fa, fb)
        prev: np.ndarray[tuple[Ps], np.dtype[F]] = fa
        next: np.ndarray[tuple[Ps], np.dtype[F]] = fb
        rest: list[np.ndarray[tuple[Ps], np.dtype[F]]] = list(fcs)

        while len(rest) > 0:
            prev = next
            next = rest.pop(0)

            cmp: np.ndarray[tuple[Ps], np.dtype[F]] = _compare(prev, next)

            if self._strict:
                # strict: all comparisons must have the same result (</=/>)
                acc[cmp != acc] = np.nan
            else:
                # weak
                # (a) move from constant to weakly monotonic
                np.copyto(acc, cmp, where=(acc == 0), casting="no")
                # (b) keep constant or weakly monotonic as-is
                # (c) comparison must not change sign (< vs >)
                acc[(cmp != 0) & (cmp != acc)] = np.nan

        return acc

    @checked_data_bounds
    @override
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> tuple[
        np_sndarray[Ps, Ns, np.dtype[F]],
        np_sndarray[Ps, Ns, np.dtype[F]],
    ]:
        a_const = not self._a.has_data
        b_const = not self._b.has_data
        cs_const = [not c.has_data for c in self._cs]
        assert not (a_const and b_const and all(cs_const)), (
            "constant monotonicity has no data bounds"
        )

        raise NotImplementedError()

    @override
    def __repr__(self) -> str:
        abc = ", ".join([repr(self._a), repr(self._b)] + [repr(c) for c in self._cs])
        return f"compare({abc}, strict={self._strict})"


@overload
def _compare(a: Fi, b: Fi) -> Fi:
    pass


@overload
def _compare(
    a: np.ndarray[S, np.dtype[F]], b: np.ndarray[S, np.dtype[F]]
) -> np.ndarray[S, np.dtype[F]]:
    pass


def _compare(a, b):
    cmp = np.full_like(a, np.nan)
    cmp[a < b] = -1
    cmp[a == b] = 0
    cmp[a > b] = +1
    return cmp
