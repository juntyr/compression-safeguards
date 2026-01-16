import sys
from collections.abc import Callable, Mapping
from warnings import warn

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import _broadcast_to, _floating_e, _floating_pi
from ....utils.bindings import Parameter
from ..typing import F, Ns, Ps, np_sndarray
from .abc import AnyExpr, EmptyExpr


class Number(EmptyExpr):
    __slots__: tuple[str, ...] = ("_n",)
    _n: str

    ZERO: "Number"
    ONE: "Number"
    TWO: "Number"

    def __init__(self, n: str) -> None:
        self._n = n

    @staticmethod
    def from_symbolic_int(n: int) -> "Number":
        try:
            return Number(f"{n}")
        except ValueError:
            pass

        warn(
            "symbolic integer evaluation, likely a**b, resulted in excessive "
            + "number of digits, try making one operand a floating-point "
            + "literal",
            category=RuntimeWarning,
        )

        int_max_str_digits = sys.get_int_max_str_digits()
        sys.set_int_max_str_digits(0)

        try:
            return Number(f"{n}")
        finally:
            sys.set_int_max_str_digits(int_max_str_digits)

    @staticmethod
    def from_symbolic_int_as_float(n: int, force_negative: bool = False) -> "Number":
        expr = Number.from_symbolic_int(n)
        if force_negative and not expr._n.startswith("-"):
            expr._n = f"-{expr._n}"
        expr._n = f"{expr._n}.0"
        return expr

    @property
    @override
    def args(self) -> tuple[()]:
        return ()

    @override
    def with_args(self) -> "Number":
        return Number(self._n)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return dtype.type(self._n)

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        n: F = Xs.dtype.type(self._n)
        return _broadcast_to(n, Xs.shape[:1])

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
        assert False, "number literals have no data bounds"

    def as_int(self) -> None | int:
        if (
            ("." in self._n)
            or ("e" in self._n)
            or ("inf" in self._n)
            or ("nan" in self._n)
        ):
            return None
        try:
            return int(self._n)
        except ValueError:
            pass

        warn(
            "symbolic integer evaluation, likely a**b, resulted in excessive "
            + "number of digits, try making one operand a floating-point "
            + "literal",
            category=RuntimeWarning,
        )

        int_max_str_digits = sys.get_int_max_str_digits()
        sys.set_int_max_str_digits(0)

        try:
            return int(self._n)
        finally:
            sys.set_int_max_str_digits(int_max_str_digits)

    @override
    def __repr__(self) -> str:
        return self._n

    @staticmethod
    def symbolic_fold_unary(expr: AnyExpr, m: Callable[[int], int]) -> "None | Number":
        if not isinstance(expr, Number):
            return None
        i = expr.as_int()
        if i is None:
            return None
        return Number.from_symbolic_int(m(i))

    @staticmethod
    def symbolic_fold_binary(
        a: AnyExpr, b: AnyExpr, m: Callable[[int, int], int]
    ) -> "None | Number":
        if not isinstance(a, Number) or not isinstance(b, Number):
            return None
        ai = a.as_int()
        bi = b.as_int()
        if (ai is None) or (bi is None):
            return None
        return Number.from_symbolic_int(m(ai, bi))


Number.ZERO = Number.from_symbolic_int(0)
Number.ONE = Number.from_symbolic_int(1)
Number.TWO = Number.from_symbolic_int(2)


class Pi(EmptyExpr):
    __slots__: tuple[str, ...] = ()

    @property
    @override
    def args(self) -> tuple[()]:
        return ()

    @override
    def with_args(self) -> "Pi":
        return Pi()

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return _floating_pi(dtype)

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        pi: F = _floating_pi(Xs.dtype)
        return _broadcast_to(pi, Xs.shape[:1])

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
        assert False, "pi has no data bounds"

    @override
    def __repr__(self) -> str:
        return "pi"


class Euler(EmptyExpr):
    __slots__: tuple[str, ...] = ()

    @property
    @override
    def args(self) -> tuple[()]:
        return ()

    @override
    def with_args(self) -> "Euler":
        return Euler()

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return _floating_e(dtype)

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        e: F = _floating_e(Xs.dtype)
        return _broadcast_to(e, Xs.shape[:1])

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
        assert False, "Euler's e has no data bounds"

    @override
    def __repr__(self) -> str:
        return "e"
