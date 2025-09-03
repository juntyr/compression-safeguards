from collections.abc import Mapping
from enum import Enum, auto
from typing import Callable

import numpy as np
from typing_extensions import assert_never  # MSPV 3.11

from ....utils._compat import _symmetric_modulo
from ....utils.bindings import Parameter
from .abc import Expr
from .addsub import ScalarSubtract
from .constfold import ScalarFoldedConstant
from .divmul import ScalarDivide, ScalarMultiply
from .group import Group
from .literal import Number
from .neg import ScalarNegate
from .typing import F, Fi, Ns, Ps, PsI


class ScalarSymmetricModulo(Expr[Expr, Expr]):
    __slots__ = ("_a", "_b")
    _a: Expr
    _b: Expr

    def __init__(self, a: Expr, b: Expr):
        self._a = a
        self._b = b

    @property
    def args(self) -> tuple[Expr, Expr]:
        return (self._a, self._b)

    def with_args(self, a: Expr, b: Expr) -> "ScalarSymmetricModulo":
        return ScalarSymmetricModulo(a, b)

    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | Expr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a,
            self._b,
            dtype,
            lambda a, b: _symmetric_modulo(a, b),  # type: ignore
            ScalarSymmetricModulo,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return _symmetric_modulo(
            self._a.eval(x, Xs, late_bound), self._b.eval(x, Xs, late_bound)
        )

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        assert False, "cannot compute the data bounds for symmetric_modulo"

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
            return (0,) + tuple(j for i in range(1, p + 1) for j in (i, -i))
        case FiniteDifference.forward:
            return tuple(i for i in range(order + accuracy))
        case FiniteDifference.backwards:
            return tuple(-i for i in range(order + accuracy))
        case _:
            assert_never(type)


def finite_difference_coefficients(
    order: int,
    offsets: tuple[Expr, ...],
    centre_dist: Callable[[Expr], Expr] = lambda x: x,
    delta_transform: Callable[[Expr], Expr] = lambda x: x,
) -> tuple[Expr, ...]:
    """
    Finite difference coefficient algorithm from:

    Fornberg, B. (1988). Generation of finite difference formulas on arbitrarily
    spaced grids. *Mathematics of Computation*, 51(184), 699-706. Available from:
    [doi:10.1090/s0025-5718-1988-0935077-0](https://doi.org/10.1090/s0025-5718-1988-0935077-0).
    """

    dx0 = centre_dist
    M: int = order
    a: tuple[Expr, ...] = offsets
    N: int = len(a) - 1

    # we explicitly the coefficient fraction into numerator and denominator
    #  expressions to delay the division for as long as possible, which allows
    #  more symbolic integer constant folding to occur
    coeffs_num: dict[tuple[int, int, int], Expr] = {
        (0, 0, 0): Number.ONE,
    }
    coeffs_denom: dict[tuple[int, int, int], Expr] = {
        (0, 0, 0): Number.ONE,
    }

    c1: Expr = Number.ONE

    for n in range(1, N + 1):
        c2: Expr = Number.ONE
        for v in range(0, n):
            c3: Expr = delta_transform(Group(ScalarSubtract(a[n], a[v])))
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
                    coeffs_denom[(m, n, v)] = ScalarMultiply(
                        ScalarMultiply(
                            coeffs_denom[(m, n - 1, v)],
                            coeffs_denom[(m - 1, n - 1, v)],
                        ),
                        c3,
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
                coeffs_denom[(m, n, n)] = ScalarMultiply(
                    ScalarMultiply(
                        coeffs_denom[(m - 1, n - 1, n - 1)],
                        coeffs_denom[(m, n - 1, n - 1)],
                    ),
                    c2,
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
