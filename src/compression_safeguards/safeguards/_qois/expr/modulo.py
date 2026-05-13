from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import _ceil_modulo, _round_ties_even_modulo
from ....utils.bindings import Parameter
from ..context import Callback, Context
from ..typing import F, Fi, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant


class ScalarFloorModulo(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_b")
    _a: AnyExpr
    _b: AnyExpr

    def __init__(self, a: AnyExpr, b: AnyExpr) -> None:
        self._a = a
        self._b = b

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._a, self._b)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarFloorModulo":
        return ScalarFloorModulo(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a,
            self._b,
            dtype,
            np.mod,
            ScalarFloorModulo,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.mod(self._a.eval(Xs, late_bound), self._b.eval(Xs, late_bound))

    @override
    def deferred_compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
        ctx: Context[Ps, Ns, F],
        callback: Callback[Ps, Ns, F],
    ) -> None:
        assert False, "cannot compute the data bounds for floor_modulo"

    @override
    def __repr__(self) -> str:
        return f"floor_modulo({self._a!r}, {self._b!r})"


class ScalarCeilModulo(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_b")
    _a: AnyExpr
    _b: AnyExpr

    def __init__(self, a: AnyExpr, b: AnyExpr) -> None:
        self._a = a
        self._b = b

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._a, self._b)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarCeilModulo":
        return ScalarCeilModulo(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a,
            self._b,
            dtype,
            _ceil_modulo,
            ScalarCeilModulo,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return _ceil_modulo(self._a.eval(Xs, late_bound), self._b.eval(Xs, late_bound))

    @override
    def deferred_compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
        ctx: Context[Ps, Ns, F],
        callback: Callback[Ps, Ns, F],
    ) -> None:
        assert False, "cannot compute the data bounds for ceil_modulo"

    @override
    def __repr__(self) -> str:
        return f"ceil_modulo({self._a!r}, {self._b!r})"


class ScalarTruncModulo(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_b")
    _a: AnyExpr
    _b: AnyExpr

    def __init__(self, a: AnyExpr, b: AnyExpr) -> None:
        self._a = a
        self._b = b

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._a, self._b)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarTruncModulo":
        return ScalarTruncModulo(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a,
            self._b,
            dtype,
            np.fmod,
            ScalarTruncModulo,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.fmod(self._a.eval(Xs, late_bound), self._b.eval(Xs, late_bound))

    @override
    def deferred_compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
        ctx: Context[Ps, Ns, F],
        callback: Callback[Ps, Ns, F],
    ) -> None:
        assert False, "cannot compute the data bounds for trunc_modulo"

    @override
    def __repr__(self) -> str:
        return f"trunc_modulo({self._a!r}, {self._b!r})"


class ScalarRoundTiesEvenModulo(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_b")
    _a: AnyExpr
    _b: AnyExpr

    def __init__(self, a: AnyExpr, b: AnyExpr) -> None:
        self._a = a
        self._b = b

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._a, self._b)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarRoundTiesEvenModulo":
        return ScalarRoundTiesEvenModulo(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a,
            self._b,
            dtype,
            _round_ties_even_modulo,
            ScalarRoundTiesEvenModulo,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return _round_ties_even_modulo(
            self._a.eval(Xs, late_bound), self._b.eval(Xs, late_bound)
        )

    @override
    def deferred_compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
        ctx: Context[Ps, Ns, F],
        callback: Callback[Ps, Ns, F],
    ) -> None:
        assert False, "cannot compute the data bounds for round_ties_even_modulo"

    @override
    def __repr__(self) -> str:
        return f"round_ties_even_modulo({self._a!r}, {self._b!r})"
