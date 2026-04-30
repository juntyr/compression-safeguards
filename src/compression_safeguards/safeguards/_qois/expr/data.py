from collections.abc import Mapping
from typing import Self
from warnings import warn

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import _broadcast_to, _ensure_array, _is_of_shape, _ones, _zeros
from ....utils.bindings import Parameter
from ....utils.error import QuantityOfInterestRuntimeWarning
from ..bound import DataBounds, data_bounds
from ..context import Callback, Context
from ..typing import F, Ns, Ps, np_sndarray
from .abc import AnyExpr, DataExpr, LateBoundConstantExpr


class Data(DataExpr):
    __slots__: tuple[str, ...] = ("_index",)
    _index: tuple[int, ...]

    SCALAR: "Data"

    def __init__(self, index: tuple[int, ...]) -> None:
        self._index = index

    @property
    def index(self) -> tuple[int, ...]:
        return self._index

    @property
    @override
    def args(self) -> tuple[()]:
        return ()

    @property
    @override
    def extra(self) -> tuple[tuple[int, ...]]:
        return (self._index,)

    @override
    def with_args(self) -> "Data":
        return Data(self._index)

    @override  # type: ignore
    def eval_has_data(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[np.bool]]:
        # we could just return all True here, since all data is ... data
        # but we also currently guarantee that NaN data values stay NaN,
        #  i.e. that the values are constant, so not really data dependent
        # therefore, we say that only non-NaN data is actually data
        data = self.eval(Xs, late_bound)
        data_is_not_nan = _ensure_array(np.isnan(data))
        np.logical_not(data_is_not_nan, out=data_is_not_nan)
        return data_is_not_nan

    @property
    @override
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return frozenset([self._index])

    @override
    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Self:
        index = list(self._index)
        index[axis] += offset
        return type(self)(index=tuple(index))

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return self

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        out: np.ndarray[tuple[int, ...], np.dtype[F]] = Xs[(...,) + self._index]
        assert _is_of_shape(out, Xs.shape[:1])
        return out

    @data_bounds(DataBounds.infallible)
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
        exprv = Xs[(...,) + self._index]

        if not np.all((expr_lower <= exprv) | np.isnan(exprv)):
            warn(
                "data lower bounds are above the data values",
                category=QuantityOfInterestRuntimeWarning,
            )
        if not np.all((expr_upper >= exprv) | np.isnan(exprv)):
            warn(
                "data upper bounds are below the data values",
                category=QuantityOfInterestRuntimeWarning,
            )

        Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(-np.inf)
        )
        Xs_lower[np.isnan(Xs)] = np.nan
        Xs_lower[(...,) + self._index] = expr_lower

        Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(np.inf)
        )
        Xs_upper[np.isnan(Xs)] = np.nan
        Xs_upper[(...,) + self._index] = expr_upper

        return callback(Xs_lower, Xs_upper)

    @override
    def __repr__(self) -> str:
        if self._index == ():
            return "x"
        return f"X[{','.join(str(i) for i in self._index)}]"


Data.SCALAR = Data(index=())


class LateBoundConstant(LateBoundConstantExpr):
    __slots__: tuple[str, ...] = ("_name", "_index")
    _name: Parameter
    _index: tuple[int, ...]

    def __init__(self, name: Parameter, index: tuple[int, ...]) -> None:
        self._name = name
        self._index = index

    @staticmethod
    def like(name: Parameter, data: Data) -> "LateBoundConstant":
        return LateBoundConstant(name, data.index)

    @property
    def name(self) -> Parameter:
        return self._name

    @property
    def index(self) -> tuple[int, ...]:
        return self._index

    @property
    @override
    def args(self) -> tuple[()]:
        return ()

    @property
    @override
    def extra(self) -> tuple[str, tuple[int, ...]]:
        return (self.name, self._index)

    @override
    def with_args(self) -> "LateBoundConstant":
        return LateBoundConstant(self._name, self._index)

    @override
    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Self:
        index = list(self._index)
        index[axis] += offset
        return type(self)(self._name, index=tuple(index))

    @property
    @override
    def late_bound_constants(self) -> frozenset[Parameter]:
        return frozenset([self.name])

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return self

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        out: np.ndarray[tuple[int, ...], np.dtype[F]] = late_bound[self.name][
            (...,) + self._index
        ]
        assert _is_of_shape(out, Xs.shape[:1])
        return out

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
        assert False, "late-bound constants have no data bounds"

    @override
    def __repr__(self) -> str:
        if self._index == ():
            return f'c["{self._name}"]'
        return f'C["{self._name}"][{",".join(str(i) for i in self._index)}]'


# scalar constant that tricks the QoIs by pretending to be data,
# which is useful for tests and fuzzing to produce specific data-like values
class ScalarAnyDataConstant(DataExpr):
    __slots__: tuple[str, ...] = ("_const",)
    _const: np.number

    def __init__(self, const: np.number) -> None:
        self._const = const

    @property
    @override
    def args(self) -> tuple[()]:
        return ()

    @property
    @override
    def extra(self) -> tuple[np.number]:
        return (self._const,)

    @override
    def with_args(self) -> "ScalarAnyDataConstant":
        return ScalarAnyDataConstant(self._const)

    @override  # type: ignore
    def eval_has_data(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[np.bool]]:
        # we could just return all True here, since all data is ... data
        # but we also currently guarantee that NaN data values stay NaN,
        #  i.e. that the values are constant, so not really data dependent
        # therefore, we say that only non-NaN data is actually data
        if np.isnan(self._const):
            return _zeros(Xs.shape[:1], dtype=np.dtype(np.bool))
        return _ones(Xs.shape[:1], dtype=np.dtype(np.bool))

    @property
    @override
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return frozenset()

    @override
    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Self:
        return self

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return self

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        assert isinstance(self._const, Xs.dtype.type)
        return _broadcast_to(self._const, Xs.shape[:1])

    @data_bounds(DataBounds.infallible)
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
        assert isinstance(self._const, Xs.dtype.type)

        if not np.all((expr_lower <= self._const) | np.isnan(self._const)):
            warn(
                "data lower bounds are above the any data constant",
                category=QuantityOfInterestRuntimeWarning,
            )
        if not np.all((expr_upper >= self._const) | np.isnan(self._const)):
            warn(
                "data upper bounds are below the any data constant",
                category=QuantityOfInterestRuntimeWarning,
            )

        Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(-np.inf)
        )
        Xs_lower[np.isnan(Xs)] = np.nan

        Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(np.inf)
        )
        Xs_upper[np.isnan(Xs)] = np.nan

        return callback(Xs_lower, Xs_upper)

    @override
    def __repr__(self) -> str:
        return f"any_data({self._const!r})"
