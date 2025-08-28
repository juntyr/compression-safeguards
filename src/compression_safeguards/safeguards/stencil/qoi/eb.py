"""
Stencil quantity of interest (QoI) error bound safeguard.
"""

__all__ = ["StencilQuantityOfInterestErrorBoundSafeguard"]

from collections.abc import Sequence, Set

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ....utils._compat import _isnan, _where
from ....utils.bindings import Bindings, Parameter
from ....utils.cast import (
    lossless_cast,
    saturating_finite_float_cast,
    to_float,
)
from ....utils.intervals import Interval, IntervalUnion
from ....utils.typing import F, S, T
from ..._qois import StencilQuantityOfInterest
from ..._qois.interval import compute_safe_data_lower_upper_interval_union
from ...eb import (
    ErrorBound,
    _apply_finite_qoi_error_bound,
    _check_error_bound,
    _compute_finite_absolute_error,
    _compute_finite_absolute_error_bound,
)
from ...qois import StencilQuantityOfInterestExpression
from .. import (
    BoundaryCondition,
    NeighbourhoodAxis,
    NeighbourhoodBoundaryAxis,
    _pad_with_boundary,
)
from ..abc import StencilSafeguard


class StencilQuantityOfInterestErrorBoundSafeguard(StencilSafeguard):
    """
    The `StencilQuantityOfInterestErrorBoundSafeguard` guarantees that the
    pointwise error `type` on a derived quantity of interest (QoI) over a
    neighbourhood of data points is less than or equal to the provided bound
    `eb`.

    The quantity of interest is specified as a non-constant expression, in
    string form, over the neighbourhood tensor `X` that is centred on the
    pointwise value `x`. For example, to bound the error on the four-neighbour
    box mean in a 3x3 neighbourhood (where `x = X[I]`), set
    `qoi="(X[I[0]-1, I[1]] + X[I[0]+1, I[1]] + X[I[0], I[1]-1] + X[I[0], I[1]+1]) / 4"`.
    Note that `X` can be indexed absolute or relative to the centred data point
    `x` using the index array `I`.

    The stencil QoI safeguard can also be used to bound the pointwise error of
    the finite-difference-approximated derivative (of arbitrary order,
    accuracy, and grid spacing) over the data by using the `finite_difference`
    function in the `qoi` expression.

    The shape of the data neighbourhood is specified as an ordered list of
    unique data axes and boundary conditions that are applied to these axes.
    If the safeguard is applied to data with an insufficient number of
    dimensions, it raises an exception. If the safeguard is applied to data
    with additional dimensions, it is indendently applied along these extra
    axes. For instance, a 2d QoI is applied to independently to all 2d slices
    in a 3d data cube.

    If the data neighbourhood uses the
    [valid][compression_safeguards.safeguards.stencil.BoundaryCondition.valid]
    boundary condition along an axis, the safeguard is only applied to data
    neighbourhoods centred on data points that have sufficient points before
    and after to satisfy the neighbourhood shape, i.e. it is not applied to
    all data points. If the axis is smaller than required by the neighbourhood
    along this axis, the safeguard is not applied. Using a different
    [`BoundaryCondition`][compression_safeguards.safeguards.stencil.BoundaryCondition]
    ensures that the safeguard is always applied to all data points.

    If the derived quantity of interest for a data neighbourhood evaluates to
    an infinite value, this safeguard guarantees that the quantity of interest
    on the decoded data neighbourhood produces the exact same infinite value.
    For a NaN quantity of interest, this safeguard guarantees that the quantity
    of interest on the decoded data neighbourhood is also NaN, but does not
    guarantee that it has the same bit pattern.

    The error bound can be verified by evaluating the QoI using the
    [`evaluate_qoi`][compression_safeguards.safeguards.stencil.qoi.eb.StencilQuantityOfInterestErrorBoundSafeguard.evaluate_qoi]
    method, which returns the the QoI in a sufficiently large floating point
    type (keeps the same dtype for floating point data, chooses a dtype with a
    mantissa that has at least as many bits as / for the integer dtype).

    Please refer to the
    [`StencilQuantityOfInterestExpression`][compression_safeguards.safeguards.qois.StencilQuantityOfInterestExpression]
    for the EBNF grammar that specifies the language in which the quantities of
    interest are written.

    The implementation of the error bound on pointwise quantities of interest
    is inspired by:

    > Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. *Proceedings of the VLDB Endowment*.
    16, 4 (December 2022), 697-710. Available from:
    [doi:10.14778/3574245.3574255](https://doi.org/10.14778/3574245.3574255).

    Parameters
    ----------
    qoi : StencilExpr
        The non-constant expression for computing the derived quantity of
        interest over a neighbourhood tensor `X`.
    neighbourhood : Sequence[dict | NeighbourhoodBoundaryAxis]
        The non-empty axes of the data neighbourhood for which the quantity of
        interest is computed. The neighbourhood window is applied independently
        over any additional axes in the data.

        The per-axis boundary conditions are applied to the data in their order
        in the neighbourhood, i.e. earlier boundary extensions can influence
        later ones.
    type : str | ErrorBound
        The type of error bound on the quantity of interest that is enforced by
        this safeguard.
    eb : int | float | str | Parameter
        The value of or late-bound parameter name for the error bound on the
        quantity of interest that is enforced by this safeguard.
    """

    __slots__ = (
        "_qoi",
        "_neighbourhood",
        "_type",
        "_eb",
        "_qoi_expr",
    )
    _qoi: StencilQuantityOfInterestExpression
    _neighbourhood: tuple[NeighbourhoodBoundaryAxis, ...]
    _type: ErrorBound
    _eb: int | float | Parameter
    _qoi_expr: StencilQuantityOfInterest

    kind = "qoi_eb_stencil"

    def __init__(
        self,
        qoi: StencilQuantityOfInterestExpression,
        neighbourhood: Sequence[dict | NeighbourhoodBoundaryAxis],
        type: str | ErrorBound,
        eb: int | float | str | Parameter,
    ) -> None:
        self._neighbourhood = tuple(
            axis
            if isinstance(axis, NeighbourhoodBoundaryAxis)
            else NeighbourhoodBoundaryAxis.from_config(axis)
            for axis in neighbourhood
        )
        assert len(self._neighbourhood) > 0, "neighbourhood must not be empty"
        assert len(set(axis.axis for axis in self._neighbourhood)) == len(
            self._neighbourhood
        ), "neighbourhood axes must be unique"

        self._type = type if isinstance(type, ErrorBound) else ErrorBound[type]

        if isinstance(eb, Parameter):
            self._eb = eb
        elif isinstance(eb, str):
            self._eb = Parameter(eb)
        else:
            _check_error_bound(self._type, eb)
            self._eb = eb

        shape = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)
        I = tuple(axis.before for axis in self._neighbourhood)  # noqa: E741

        try:
            qoi_expr = StencilQuantityOfInterest(qoi, stencil_shape=shape, stencil_I=I)
        except Exception as err:
            raise AssertionError(
                f"failed to parse stencil QoI expression {qoi!r}: {err}"
            ) from err

        self._qoi = qoi
        self._qoi_expr = qoi_expr

    @property
    def late_bound(self) -> Set[Parameter]:
        """
        The set of late-bound parameters that this safeguard has.

        Late-bound parameters are only bound when checking and applying the
        safeguard, in contrast to the normal early-bound parameters that are
        configured during safeguard initialisation.

        Late-bound parameters can be used for parameters that depend on the
        specific data that is to be safeguarded.
        """

        parameters = set(self._qoi_expr.late_bound_constants)

        if isinstance(self._eb, Parameter):
            parameters.add(self._eb)

        for axis in self._neighbourhood:
            if isinstance(axis.constant_boundary, Parameter):
                parameters.add(axis.constant_boundary)

        return frozenset(parameters)

    def compute_check_neighbourhood_for_data_shape(
        self,
        data_shape: tuple[int, ...],
    ) -> tuple[None | NeighbourhoodAxis, ...]:
        """
        Compute the shape of the data neighbourhood for data of a given shape.
        [`None`][None] is returned along dimensions for which the stencil QoI
        safeguard does not need to look at adjacent data points.

        This method also checks that the data shape is compatible with the
        stencil QoI safeguard.

        Parameters
        ----------
        data_shape : tuple[int, ...]
            The shape of the data.

        Returns
        -------
        neighbourhood_shape : tuple[None | NeighbourhoodAxis, ...]
            The shape of the data neighbourhood.
        """

        neighbourhood: list[None | NeighbourhoodAxis] = [None] * len(data_shape)

        all_axes = []
        for axis in self._neighbourhood:
            if (axis.axis >= len(data_shape)) or (axis.axis < -len(data_shape)):
                raise IndexError(
                    f"axis index {axis.axis} is out of bounds for array of shape {data_shape}"
                )
            naxis = axis.axis if axis.axis >= 0 else len(data_shape) + axis.axis
            if naxis in all_axes:
                raise IndexError(
                    f"duplicate axis index {axis.axis}, normalised to {naxis}, for array of shape {data_shape}"
                )
            all_axes.append(naxis)

            neighbourhood[naxis] = axis.shape

        if np.prod(data_shape) == 0:
            return (None,) * len(data_shape)

        return tuple(neighbourhood)

    def evaluate_qoi(
        self,
        data: np.ndarray[S, np.dtype[T]],
        late_bound: Bindings,
    ) -> np.ndarray[tuple[int, ...], np.dtype[F]]:
        """
        Evaluate the derived quantity of interest on the `data`.

        The quantity of interest may have a different shape if the
        [valid][compression_safeguards.safeguards.stencil.BoundaryCondition.valid]
        boundary condition is used along any axis.

        If the `data` is of integer dtype, the quantity of interest is
        evaluated in floating point with sufficient precision to represent all
        integer values.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Data for which the quantity of interest is evaluated.
        late_bound : Bindings
            Bindings for late-bound constants in the quantity of interest.

        Returns
        -------
        qoi : np.ndarray[tuple[int, ...], np.dtype[F]]
            Evaluated quantity of interest, in floating point.
        """

        # check that the data shape is compatible with the neighbourhood shape
        self.compute_check_neighbourhood_for_data_shape(data.shape)

        empty_shape = list(data.shape)

        # if the neighbourhood is empty, e.g. because we in valid mode and the
        #  neighbourhood shape exceeds the data shape, return empty
        for axis in self._neighbourhood:
            if axis.boundary == BoundaryCondition.valid:
                empty_shape[axis.axis] = max(
                    0, empty_shape[axis.axis] - axis.before - axis.after
                )

        if any(s == 0 for s in empty_shape):
            float_dtype: np.dtype[F] = to_float(
                np.array([0, 1], dtype=data.dtype)
            ).dtype
            return np.zeros(empty_shape, dtype=float_dtype)

        constant_boundaries = [
            None
            if axis.constant_boundary is None
            else late_bound.resolve_ndarray_with_lossless_cast(
                axis.constant_boundary, (), data.dtype
            )
            if isinstance(axis.constant_boundary, Parameter)
            else lossless_cast(
                axis.constant_boundary,
                data.dtype,
                "stencil QoI safeguard constant boundary",
            )
            for axis in self._neighbourhood
        ]

        data_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = data
        for axis, axis_constant_boundary in zip(
            self._neighbourhood, constant_boundaries
        ):
            data_boundary = _pad_with_boundary(
                data_boundary,
                axis.boundary,
                axis.before,
                axis.after,
                axis_constant_boundary,
                axis.axis,
            )

        window = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)

        data_windows_float: np.ndarray[tuple[int, ...], np.dtype[F]] = to_float(
            sliding_window_view(
                data_boundary,
                window,
                axis=tuple(axis.axis for axis in self._neighbourhood),
                writeable=False,
            )  # type: ignore
        )

        late_bound_constants: dict[
            Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]
        ] = dict()
        for c in self._qoi_expr.late_bound_constants:
            late_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = (
                late_bound.resolve_ndarray_with_lossless_cast(c, data.shape, data.dtype)
            )
            for axis, axis_constant_boundary in zip(
                self._neighbourhood, constant_boundaries
            ):
                late_boundary = _pad_with_boundary(
                    late_boundary,
                    axis.boundary,
                    axis.before,
                    axis.after,
                    axis_constant_boundary,
                    axis.axis,
                )
            late_windows_float: np.ndarray[tuple[int, ...], np.dtype[F]] = to_float(
                sliding_window_view(
                    late_boundary,
                    window,
                    axis=tuple(axis.axis for axis in self._neighbourhood),
                    writeable=False,
                )  # type: ignore
            )
            late_bound_constants[c] = late_windows_float

        return self._qoi_expr.eval(
            data_windows_float,
            late_bound_constants,
        )

    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        decoded: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements in the `decoded` array satisfy the error bound for
        the quantity of interest over a neighbourhood on the `data`.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Data to be encoded.
        decoded : np.ndarray[S, np.dtype[T]]
            Decoded data.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        ok : np.ndarray[S, np.dtype[np.bool]]
            Pointwise, `True` if the check succeeded for this element.
        """

        # check that the data shape is compatible with the neighbourhood shape
        self.compute_check_neighbourhood_for_data_shape(data.shape)

        window = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)

        # if the neighbourhood is empty, e.g. because we in valid mode and the
        #  neighbourhood shape exceeds the data shape, allow all values
        for axis, w in zip(self._neighbourhood, window):
            if data.shape[axis.axis] < w and axis.boundary == BoundaryCondition.valid:
                return np.ones_like(data, dtype=np.bool)  # type: ignore
        if data.size == 0:
            return np.ones_like(data, dtype=np.bool)  # type: ignore

        constant_boundaries = [
            None
            if axis.constant_boundary is None
            else late_bound.resolve_ndarray_with_lossless_cast(
                axis.constant_boundary, (), data.dtype
            )
            if isinstance(axis.constant_boundary, Parameter)
            else lossless_cast(
                axis.constant_boundary,
                data.dtype,
                "stencil QoI safeguard constant boundary",
            )
            for axis in self._neighbourhood
        ]

        data_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = data
        decoded_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = decoded
        for axis, axis_constant_boundary in zip(
            self._neighbourhood, constant_boundaries
        ):
            data_boundary = _pad_with_boundary(
                data_boundary,
                axis.boundary,
                axis.before,
                axis.after,
                axis_constant_boundary,
                axis.axis,
            )
            decoded_boundary = _pad_with_boundary(
                decoded_boundary,
                axis.boundary,
                axis.before,
                axis.after,
                axis_constant_boundary,
                axis.axis,
            )

        data_windows_float: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
            to_float(
                sliding_window_view(
                    data_boundary,
                    window,
                    axis=tuple(axis.axis for axis in self._neighbourhood),
                    writeable=False,
                )  # type: ignore
            )
        )
        decoded_windows_float: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
            to_float(
                sliding_window_view(
                    decoded_boundary,
                    window,
                    axis=tuple(axis.axis for axis in self._neighbourhood),
                    writeable=False,
                )  # type: ignore
            )
        )

        late_bound_constants: dict[
            Parameter, np.ndarray[tuple[int, ...], np.dtype[np.floating]]
        ] = dict()
        for c in self._qoi_expr.late_bound_constants:
            late_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = (
                late_bound.resolve_ndarray_with_lossless_cast(c, data.shape, data.dtype)
            )
            for axis, axis_constant_boundary in zip(
                self._neighbourhood, constant_boundaries
            ):
                late_boundary = _pad_with_boundary(
                    late_boundary,
                    axis.boundary,
                    axis.before,
                    axis.after,
                    axis_constant_boundary,
                    axis.axis,
                )
            late_windows_float: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
                to_float(
                    sliding_window_view(
                        late_boundary,
                        window,
                        axis=tuple(axis.axis for axis in self._neighbourhood),
                        writeable=False,
                    )  # type: ignore
                )
            )
            late_bound_constants[c] = late_windows_float

        qoi_data: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
            self._qoi_expr.eval(
                data_windows_float,
                late_bound_constants,
            )
        )
        qoi_decoded: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
            self._qoi_expr.eval(
                decoded_windows_float,
                late_bound_constants,
            )
        )

        eb: np.ndarray[tuple[()] | tuple[int, ...], np.dtype[np.floating]] = (
            late_bound.resolve_ndarray_with_saturating_finite_float_cast(
                self._eb,
                qoi_data.shape,
                qoi_data.dtype,
            )
            if isinstance(self._eb, Parameter)
            else saturating_finite_float_cast(
                self._eb, qoi_data.dtype, "stencil QoI error bound safeguard eb"
            )
        )
        _check_error_bound(self._type, eb)

        finite_ok = _compute_finite_absolute_error(
            self._type, qoi_data, qoi_decoded
        ) <= _compute_finite_absolute_error_bound(self._type, eb, qoi_data)

        windows_ok: np.ndarray[tuple[int, ...], np.dtype[np.bool]] = _where(
            _isnan(qoi_data), _isnan(qoi_decoded), finite_ok | (qoi_data == qoi_decoded)
        )

        s = [slice(None)] * data.ndim
        for axis in self._neighbourhood:
            if axis.boundary == BoundaryCondition.valid:
                start = None if axis.before == 0 else axis.before
                end = None if axis.after == 0 else -axis.after
                s[axis.axis] = slice(start, end)

        ok: np.ndarray[S, np.dtype[np.bool]] = np.ones_like(data, dtype=np.bool)  # type: ignore
        ok[tuple(s)] = windows_ok

        return ok

    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the error bound is upheld with respect
        to the quantity of interest over a neighbourhood on the `data`.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Data for which the safe intervals should be computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the error bound is upheld.
        """

        # check that the data shape is compatible with the neighbourhood shape
        self.compute_check_neighbourhood_for_data_shape(data.shape)

        window = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)

        # if the neighbourhood is empty, e.g. because we in valid mode and the
        #  neighbourhood shape exceeds the data shape, allow all values
        for axis, w in zip(self._neighbourhood, window):
            if data.shape[axis.axis] < w and axis.boundary == BoundaryCondition.valid:
                return Interval.full_like(data).into_union()
        if data.size == 0:
            return Interval.full_like(data).into_union()

        constant_boundaries = [
            None
            if axis.constant_boundary is None
            else late_bound.resolve_ndarray_with_lossless_cast(
                axis.constant_boundary, (), data.dtype
            )
            if isinstance(axis.constant_boundary, Parameter)
            else lossless_cast(
                axis.constant_boundary,
                data.dtype,
                "stencil QoI safeguard constant boundary",
            )
            for axis in self._neighbourhood
        ]

        data_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = data
        for axis, axis_constant_boundary in zip(
            self._neighbourhood, constant_boundaries
        ):
            data_boundary = _pad_with_boundary(
                data_boundary,
                axis.boundary,
                axis.before,
                axis.after,
                axis_constant_boundary,
                axis.axis,
            )

        data_windows_float: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
            to_float(
                sliding_window_view(
                    data_boundary,
                    window,
                    axis=tuple(axis.axis for axis in self._neighbourhood),
                    writeable=False,
                )  # type: ignore
            )
        )

        late_bound_constants: dict[
            Parameter, np.ndarray[tuple[int, ...], np.dtype[np.floating]]
        ] = dict()
        for c in self._qoi_expr.late_bound_constants:
            late_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = (
                late_bound.resolve_ndarray_with_lossless_cast(c, data.shape, data.dtype)
            )
            for axis, axis_constant_boundary in zip(
                self._neighbourhood, constant_boundaries
            ):
                late_boundary = _pad_with_boundary(
                    late_boundary,
                    axis.boundary,
                    axis.before,
                    axis.after,
                    axis_constant_boundary,
                    axis.axis,
                )
            late_windows_float: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
                to_float(
                    sliding_window_view(
                        late_boundary,
                        window,
                        axis=tuple(axis.axis for axis in self._neighbourhood),
                        writeable=False,
                    )  # type: ignore
                )
            )
            late_bound_constants[c] = late_windows_float

        data_qoi: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
            self._qoi_expr.eval(
                data_windows_float,
                late_bound_constants,
            )
        )

        eb: np.ndarray[tuple[()] | tuple[int, ...], np.dtype[np.floating]] = (
            late_bound.resolve_ndarray_with_saturating_finite_float_cast(
                self._eb,
                data_qoi.shape,
                data_qoi.dtype,
            )
            if isinstance(self._eb, Parameter)
            else saturating_finite_float_cast(
                self._eb, data_qoi.dtype, "stencil QoI error bound safeguard eb"
            )
        )
        _check_error_bound(self._type, eb)

        qoi_lower_upper: tuple[
            np.ndarray[tuple[int, ...], np.dtype[np.floating]],
            np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        ] = _apply_finite_qoi_error_bound(
            self._type,
            eb,
            data_qoi,
        )
        qoi_lower, qoi_upper = qoi_lower_upper

        # compute the bounds in data space
        data_windows_float_lower, data_windows_float_upper = (
            self._qoi_expr.compute_data_bounds(
                qoi_lower,
                qoi_upper,
                data_windows_float,
                late_bound_constants,
            )
        )

        # compute how the data indices are distributed into windows
        # i.e. for each QoI element, which data does it depend on
        indices_boundary = np.arange(data.size).reshape(data.shape)
        for axis in self._neighbourhood:
            indices_boundary = _pad_with_boundary(
                indices_boundary,
                axis.boundary,
                axis.before,
                axis.after,
                None if axis.constant_boundary is None else np.array(data.size),  # type: ignore
                axis.axis,
            )
        indices_windows = sliding_window_view(  # type: ignore
            indices_boundary,
            window,
            axis=tuple(axis.axis for axis in self._neighbourhood),
            writeable=False,
        ).reshape((-1, np.prod(window)))

        # only contribute window elements that are used in the QoI
        window_used = np.zeros(window, dtype=bool)
        for idxs in self._qoi_expr.data_indices:
            window_used[idxs] = True

        # compute the reverse: for each data element, which windows is it in
        # i.e. for each data element, which QoI elements does it contribute to
        #      and thus which error bounds affect it
        reverse_indices_windows = np.full(
            (data.size, np.sum(window_used)), indices_windows.size
        )
        reverse_indices_counter = np.zeros(data.size, dtype=int)
        for i, u in enumerate(window_used.flat):
            # skip window indices that are not used in the QoI
            if not u:
                continue
            # manual loop to account for potential aliasing:
            # with a wrapping boundary, more than one j for the same window
            # position j could refer back to the same data element
            for j in range(indices_windows.shape[0]):
                idx = indices_windows[j, i]
                if idx != data.size:
                    # lazily allocate more to account for all possible edge cases
                    if reverse_indices_counter[idx] >= reverse_indices_windows.shape[1]:
                        new_reverse_indices_windows = np.full(
                            (data.size, reverse_indices_windows.shape[1] * 2),
                            indices_windows.size,
                        )
                        new_reverse_indices_windows[
                            :, : reverse_indices_windows.shape[1]
                        ] = reverse_indices_windows
                        reverse_indices_windows = new_reverse_indices_windows
                    # update the reverse mapping
                    reverse_indices_windows[idx][reverse_indices_counter[idx]] = (
                        j * window_used.size
                    ) + i
                    reverse_indices_counter[idx] += 1

        data_float: np.ndarray[S, np.dtype[np.floating]] = to_float(data)

        # flatten the QoI data bounds and append an infinite value,
        # which is indexed if an element did not contribute to the maximum
        # number of windows
        with np.errstate(invalid="ignore"):
            data_windows_float_lower_flat = np.full(
                data_windows_float_lower.size + 1, -np.inf, data_float.dtype
            )
            data_windows_float_lower_flat[:-1] = data_windows_float_lower.flatten()
            data_windows_float_upper_flat = np.full(
                data_windows_float_upper.size + 1, np.inf, data_float.dtype
            )
            data_windows_float_upper_flat[:-1] = data_windows_float_upper.flatten()

        # for each data element, reduce over the data bounds that affect it
        # since some data elements may have no data bounds that affect them,
        #  e.g. because of the valid boundary condition, they may have infinite
        #  bounds
        data_float_lower: np.ndarray[S, np.dtype[np.floating]] = np.amax(
            data_windows_float_lower_flat[reverse_indices_windows], axis=1
        ).reshape(data.shape)
        data_float_upper: np.ndarray[S, np.dtype[np.floating]] = np.amin(
            data_windows_float_upper_flat[reverse_indices_windows], axis=1
        ).reshape(data.shape)

        return compute_safe_data_lower_upper_interval_union(
            data, data_float_lower, data_float_upper
        )

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(
            kind=type(self).kind,
            qoi=self._qoi,
            neighbourhood=[axis.get_config() for axis in self._neighbourhood],
            type=self._type.name,
            eb=self._eb,
        )
