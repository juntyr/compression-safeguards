from collections.abc import Callable, Mapping
from enum import Enum, auto
from typing import assert_never

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import _symmetric_modulo
from ....utils.bindings import Parameter
from ..typing import F, Fi, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .addsub import ScalarSubtract
from .constfold import ScalarFoldedConstant
from .divmul import ScalarDivide, ScalarMultiply
from .group import Group
from .literal import Number
from .neg import ScalarNegate


class ScalarSymmetricModulo(Expr[AnyExpr, AnyExpr]):
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

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarSymmetricModulo":
        return ScalarSymmetricModulo(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a,
            self._b,
            dtype,
            lambda a, b: _symmetric_modulo(a, b),
            ScalarSymmetricModulo,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return _symmetric_modulo(
            self._a.eval(Xs, late_bound), self._b.eval(Xs, late_bound)
        )

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
        assert False, "cannot compute the data bounds for symmetric_modulo"

    @override
    def __repr__(self) -> str:
        return f"symmetric_modulo({self._a!r}, {self._b!r})"


class FiniteDifference(Enum):
    """
    Different types of finite differences.
    """

    central = auto()
    r"""
    Central finite difference, computed over the indices
    $\{i-k; \ldots; i; \ldots; i+k\}$.
    """

    forward = auto()
    r"""
    Forward finite difference, computed over the indices $\{i; \ldots; i+k\}$.
    """

    backwards = auto()
    r"""
    Backward finite difference, computed over the indices $\{i-k; \ldots; i\}$.
    """


def finite_difference_offsets(
    type: FiniteDifference,
    order: int,
    accuracy: int,
) -> tuple[int, ...]:
    match type:
        case FiniteDifference.central:
            noffsets = order + (order % 2) - 1 + accuracy
            p = (noffsets - 1) // 2
            return (0, *tuple(j for i in range(1, p + 1) for j in (i, -i)))
        case FiniteDifference.forward:
            return tuple(i for i in range(order + accuracy))
        case FiniteDifference.backwards:
            return tuple(-i for i in range(order + accuracy))
        case _:
            assert_never(type)


def finite_difference_coefficients(
    order: int,
    offsets: tuple[AnyExpr, ...],
    centre_dist: Callable[[AnyExpr], AnyExpr] = lambda x: x,
    delta_transform: Callable[[AnyExpr], AnyExpr] = lambda x: x,
) -> tuple[AnyExpr, ...]:
    """
    Finite difference coefficient algorithm from:

    Fornberg, B. (1988). Generation of finite difference formulas on arbitrarily
    spaced grids. *Mathematics of Computation*, 51(184), 699-706. Available from:
    [doi:10.1090/s0025-5718-1988-0935077-0](https://doi.org/10.1090/s0025-5718-1988-0935077-0).
    """

    dx0 = centre_dist
    M: int = order
    a: tuple[AnyExpr, ...] = offsets
    N: int = len(a) - 1

    # we explicitly the coefficient fraction into numerator and denominator
    #  expressions to delay the division for as long as possible, which allows
    #  more symbolic integer constant folding to occur
    coeffs_num: dict[tuple[int, int, int], AnyExpr] = {
        (0, 0, 0): Number.ONE,
    }
    coeffs_denom: dict[tuple[int, int, int], AnyExpr] = {
        (0, 0, 0): Number.ONE,
    }

    c1: AnyExpr = Number.ONE

    for n in range(1, N + 1):
        c2: AnyExpr = Number.ONE
        for v in range(0, n):
            c3: AnyExpr = delta_transform(Group(ScalarSubtract(a[n], a[v])))
            c2 = Group(ScalarMultiply(c2, c3))
            if n <= M:
                coeffs_num[(n, n - 1, v)] = Number.ZERO
                coeffs_denom[(n, n - 1, v)] = Number.ONE
            for m in range(0, min(n, M) + 1):
                if m > 0:
                    coeffs_num[(m, n, v)] = Group(
                        ScalarSubtract(
                            Group(
                                ScalarMultiply(
                                    ScalarMultiply(
                                        delta_transform(dx0(a[n])),
                                        coeffs_num[(m, n - 1, v)],
                                    ),
                                    coeffs_denom[(m - 1, n - 1, v)],
                                )
                            ),
                            Group(
                                ScalarMultiply(
                                    ScalarMultiply(
                                        Number.from_symbolic_int(m),
                                        coeffs_num[(m - 1, n - 1, v)],
                                    ),
                                    coeffs_denom[(m, n - 1, v)],
                                ),
                            ),
                        )
                    )
                    coeffs_denom[(m, n, v)] = Group(
                        ScalarMultiply(
                            ScalarMultiply(
                                coeffs_denom[(m, n - 1, v)],
                                coeffs_denom[(m - 1, n - 1, v)],
                            ),
                            c3,
                        )
                    )
                else:
                    coeffs_num[(m, n, v)] = Group(
                        ScalarMultiply(
                            delta_transform(dx0(a[n])), coeffs_num[(m, n - 1, v)]
                        )
                    )
                    coeffs_denom[(m, n, v)] = Group(
                        ScalarMultiply(coeffs_denom[(m, n - 1, v)], c3)
                    )
        for m in range(0, min(n, M) + 1):
            if m > 0:
                coeffs_num[(m, n, n)] = Group(
                    ScalarMultiply(
                        c1,
                        Group(
                            ScalarSubtract(
                                Group(
                                    ScalarMultiply(
                                        ScalarMultiply(
                                            Number.from_symbolic_int(m),
                                            coeffs_num[(m - 1, n - 1, n - 1)],
                                        ),
                                        coeffs_denom[(m, n - 1, n - 1)],
                                    )
                                ),
                                Group(
                                    ScalarMultiply(
                                        ScalarMultiply(
                                            delta_transform(dx0(a[n - 1])),
                                            coeffs_num[(m, n - 1, n - 1)],
                                        ),
                                        coeffs_denom[(m - 1, n - 1, n - 1)],
                                    )
                                ),
                            )
                        ),
                    )
                )
                coeffs_denom[(m, n, n)] = Group(
                    ScalarMultiply(
                        ScalarMultiply(
                            coeffs_denom[(m - 1, n - 1, n - 1)],
                            coeffs_denom[(m, n - 1, n - 1)],
                        ),
                        c2,
                    )
                )
            else:
                coeffs_num[(m, n, n)] = Group(
                    ScalarMultiply(
                        ScalarNegate(c1),
                        ScalarMultiply(
                            delta_transform(dx0(a[n - 1])),
                            coeffs_num[(m, n - 1, n - 1)],
                        ),
                    )
                )
                coeffs_denom[(m, n, n)] = Group(
                    ScalarMultiply(c2, coeffs_denom[(m, n - 1, n - 1)])
                )
        c1 = c2

    return tuple(
        Group(ScalarDivide(coeffs_num[M, N, v], coeffs_denom[M, N, v]))
        for v in range(0, N + 1)
    )
