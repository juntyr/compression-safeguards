import sys
from collections.abc import Mapping
from typing import Callable
from warnings import warn

import numpy as np

from ....utils._compat import _broadcast_to, _e, _pi
from ....utils.bindings import Parameter
from .abc import Expr
from .typing import F, Ns, Ps, PsI


class Number(Expr):
    __slots__ = ("_n",)
    _n: str

    def __init__(self, n: str):
        self._n = n

    ZERO: "Number"
    ONE: "Number"
    TWO: "Number"

    @staticmethod
    def from_symbolic_int(n: int) -> "Number":
        try:
            return Number(f"{n}")
        except ValueError:
            pass

        warn(
            "symbolic integer evaluation, likely a**b, resulted in excessive number of digits, try making one operand a floating point literal"
        )

        int_max_str_digits = None

        if (
            getattr(sys, "get_int_max_str_digits", None) is not None
            and getattr(sys, "set_int_max_str_digits", None) is not None
        ):  # MSPV 3.11
            int_max_str_digits = sys.get_int_max_str_digits()
            sys.set_int_max_str_digits(0)

        try:
            return Number(f"{n}")
        finally:
            if int_max_str_digits is not None:
                sys.set_int_max_str_digits(int_max_str_digits)

    @staticmethod
    def from_symbolic_int_as_float(n: int, force_negative: bool = False) -> "Number":
        expr = Number.from_symbolic_int(n)
        if force_negative and not expr._n.startswith("-"):
            expr._n = f"-{expr._n}"
        expr._n = f"{expr._n}.0"
        return expr

    @property
    def args(self) -> tuple[()]:
        return ()

    def with_args(self) -> "Number":
        return Number(self._n)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return dtype.type(self._n)

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        n: F = Xs.dtype.type(self._n)
        return _broadcast_to(n, x)

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
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
            "symbolic integer evaluation, likely a**b, resulted in excessive number of digits, try making one operand a floating point literal"
        )

        int_max_str_digits = None

        if (
            getattr(sys, "get_int_max_str_digits", None) is not None
            and getattr(sys, "set_int_max_str_digits", None) is not None
        ):  # MSPV 3.11
            int_max_str_digits = sys.get_int_max_str_digits()
            sys.set_int_max_str_digits(0)

        try:
            return int(self._n)
        finally:
            if int_max_str_digits is not None:
                sys.set_int_max_str_digits(int_max_str_digits)

    def __repr__(self) -> str:
        return self._n

    @staticmethod
    def symbolic_fold_unary(expr: Expr, m: Callable[[int], int]) -> None | Expr:
        if not isinstance(expr, Number):
            return None
        i = expr.as_int()
        if i is None:
            return None
        return Number.from_symbolic_int(m(i))

    @staticmethod
    def symbolic_fold_binary(
        a: Expr, b: Expr, m: Callable[[int, int], int]
    ) -> None | Expr:
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


class Pi(Expr):
    __slots__ = ()

    @property
    def args(self) -> tuple[()]:
        return ()

    def with_args(self) -> "Pi":
        return Pi()

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return _pi(dtype)

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        pi: F = _pi(Xs.dtype)
        return _broadcast_to(pi, x)

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        assert False, "pi has no data bounds"

    def __repr__(self) -> str:
        return "pi"


class Euler(Expr):
    __slots__ = ()

    @property
    def args(self) -> tuple[()]:
        return ()

    def with_args(self) -> "Euler":
        return Euler()

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return _e(dtype)

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        e: F = _e(Xs.dtype)
        return _broadcast_to(e, x)

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        assert False, "Euler's e has no data bounds"

    def __repr__(self) -> str:
        return "e"
