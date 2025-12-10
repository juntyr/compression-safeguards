import itertools
from collections.abc import Mapping
from contextlib import contextmanager
from hashlib import blake2b
from typing import Literal

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import _broadcast_to, _ensure_array, _ones, _zeros
from ....utils.bindings import Bindings, Parameter
from ....utils.cast import from_float
from ....utils.error import ctx
from ....utils.intervals import Interval, IntervalUnion
from ....utils.typing import S, T
from ..bound import DataBounds, data_bounds
from ..typing import F, Ns, Ps, np_sndarray
from .abc import AnyExpr, EmptyExpr


class HashingExpr(EmptyExpr):
    __slots__: tuple[str, ...] = ("_data_indices", "_late_bound_constants")
    _data_indices: frozenset[tuple[int, ...]]
    _late_bound_constants: frozenset[Parameter]

    def __init__(
        self,
        data_indices: frozenset[tuple[int, ...]],
        late_bound_constants: frozenset[Parameter],
    ) -> None:
        self._data_indices = data_indices
        self._late_bound_constants = late_bound_constants

    @staticmethod
    def from_data_shape(
        data_shape: tuple[int, ...], late_bound_constants: frozenset[Parameter]
    ) -> "HashingExpr":
        data_indices = frozenset(
            [tuple(i) for i in itertools.product(*[range(a) for a in data_shape])]
        )

        return HashingExpr(data_indices, late_bound_constants)

    @property
    @override
    def args(self) -> tuple[()]:
        return ()

    @override
    def with_args(self) -> "HashingExpr":
        return HashingExpr(self._data_indices, self._late_bound_constants)

    @property  # type: ignore
    @override
    def has_data(self) -> bool:
        return True

    @override  # type: ignore
    def eval_has_data(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[np.bool]]:
        return _ones(Xs.shape[:1], dtype=np.dtype(np.bool))

    @property  # type: ignore
    @override
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return self._data_indices

    @override  # type: ignore
    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> "HashingExpr":
        data_indices = []
        for index_ in self._data_indices:
            index = list(index_)
            index[axis] += offset
            data_indices.append(tuple(index))
        return HashingExpr(
            data_indices=frozenset(data_indices),
            late_bound_constants=self._late_bound_constants,
        )

    @property  # type: ignore
    @override
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._late_bound_constants

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return self

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        x_size: Ps = Xs.shape[0]

        Xs_flat = Xs.reshape((x_size, Xs.size // x_size))
        late_bound_flat = {
            param: value.reshape((x_size, Xs.size // x_size))
            for param, value in late_bound.items()
        }

        hash: np.ndarray[tuple[Ps], np.dtype[F]] = np.empty(x_size, dtype=Xs.dtype)

        # hash Xs and late_bound stencils for every element in X
        for i in range(x_size):
            hasher = blake2b(digest_size=Xs.dtype.itemsize)
            hasher.update(Xs_flat[i].tobytes())
            for param, value_flat in sorted(
                late_bound_flat.items(), key=lambda pv: pv[0]
            ):
                hasher.update(param.encode())
                hasher.update(value_flat[i].tobytes())
            hash[i] = np.frombuffer(hasher.digest(), dtype=Xs.dtype, count=1)[0]

        return hash

    @data_bounds(DataBounds.infallible)
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
        X_hash: np.ndarray[tuple[Ps], np.dtype[F]] = self.eval(Xs, late_bound)
        Xs_hash: np_sndarray[Ps, Ns, np.dtype[F]] = _broadcast_to(
            X_hash.reshape(Xs.shape[:1] + (1,) * (Xs.ndim - 1)), Xs.shape
        )
        return _ensure_array(Xs_hash, copy=True), _ensure_array(Xs_hash, copy=True)

    @override
    def __repr__(self) -> str:
        return "hashing"


@contextmanager
def _patch_for_hashing_qoi_dev_only():
    from ...stencil.qoi import eb  # noqa: PLC0415

    old_interval_union_contains = IntervalUnion.contains
    old_interval_union_pick = IntervalUnion.pick

    old_stencil_qoi_check_pointwise = (
        eb.StencilQuantityOfInterestErrorBoundSafeguard.check_pointwise
    )
    old_compute_safe_data_lower_upper_interval_union = (
        eb.compute_safe_data_lower_upper_interval_union
    )

    IntervalUnion.contains = _interval_union_contains
    IntervalUnion.pick = _interval_union_pick

    eb.StencilQuantityOfInterestErrorBoundSafeguard.check_pointwise = (
        _stencil_qoi_check_pointwise
    )
    eb.compute_safe_data_lower_upper_interval_union = (
        _compute_safe_data_lower_upper_interval_union
    )

    try:
        yield
    finally:
        IntervalUnion.contains = old_interval_union_contains
        IntervalUnion.pick = old_interval_union_pick

        eb.StencilQuantityOfInterestErrorBoundSafeguard.check_pointwise = (
            old_stencil_qoi_check_pointwise
        )
        eb.compute_safe_data_lower_upper_interval_union = (
            old_compute_safe_data_lower_upper_interval_union
        )


def _interval_union_contains(
    self, other: np.ndarray[S, np.dtype[T]]
) -> np.ndarray[S, np.dtype[np.bool]]:
    return _ones(other.shape, dtype=np.dtype(np.bool))


def _interval_union_pick(
    self, prediction: np.ndarray[S, np.dtype[T]]
) -> np.ndarray[S, np.dtype[T]]:
    if np.all(self._lower == 1):
        raise ValueError("fuzzer hash is all ones") | ctx
    return _ensure_array(self._lower[0].reshape(prediction.shape), copy=True)


def _stencil_qoi_check_pointwise(
    self,
    data: np.ndarray[S, np.dtype[T]],
    prediction: np.ndarray[S, np.dtype[T]],
    *,
    late_bound: Bindings,
    where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
) -> np.ndarray[S, np.dtype[np.bool]]:
    if np.all(prediction == 1):
        return _zeros(data.shape, dtype=np.dtype(np.bool))
    return _ones(data.shape, dtype=np.dtype(np.bool))


def _compute_safe_data_lower_upper_interval_union(
    data: np.ndarray[S, np.dtype[T]],
    data_float_lower: np.ndarray[S, np.dtype[F]],
    data_float_upper: np.ndarray[S, np.dtype[F]],
) -> IntervalUnion[T, int, int]:
    valid = Interval.empty_like(data)
    valid._lower[:] = from_float(data_float_lower, data.dtype).flatten()
    valid._upper[:] = from_float(data_float_lower, data.dtype).flatten()
    return valid.into_union()
