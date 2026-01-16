"""
Stencil quantity of interest (QoI) error bound safeguard.
"""

__all__ = ["StencilQuantityOfInterestErrorBoundSafeguard"]

from collections.abc import Sequence, Set
from functools import reduce
from typing import ClassVar, Literal

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _ensure_array,
    _ones,
    _place,
    _reshape,
    _sliding_window_view,
    _zeros,
)
from ....utils.bindings import Bindings, Parameter
from ....utils.cast import (
    ToFloatMode,
    lossless_cast,
    saturating_finite_float_cast,
    to_float,
)
from ....utils.error import TypeCheckError, ctx, lookup_enum_or_raise
from ....utils.intervals import Interval, IntervalUnion
from ....utils.typing import JSON, F, S, T
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
    _reverse_neighbourhood_indices,
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

    /// details | Tip: Finite Differences
        type: tip
    The stencil QoI safeguard can also be used to bound the pointwise error of
    the finite-difference-approximated derivative (of arbitrary order,
    accuracy, and grid spacing) over the data by using the `finite_difference`
    function in the `qoi` expression.
    ///

    /// details | Tip: Monotonic Sequences
        type: tip
    The stencil QoI safeguard can also be used to preserve the monotonicity of
    a sequence of values, i.e. to guarantee that a sequence that was originally
    strictly/weakly monotonically increasing/decreasing/constant still is. The
    sequence can be arbitrary within the stencil neighbourhood, e.g. along a
    single axis, in a zigzag, etc. Preserving the monotonicity of multiple
    sequences, e.g. along several axes, requires multiple stencil QoI
    safeguards. For instance, to guarantee that strictly increasing/decreasing
    sequences along a single axis stay strictly increasing/decreasing, use the
    following `qoi` expression with an absolute error bound of zero (more
    monotonicity QoIs, including strict vs weak monotonicity and constant
    sequences, can be found in [test_monotonicity.py]):

    [test_monotonicity.py]: https://github.com/juntyr/compression-safeguards/blob/main/tests/test_monotonicity.py

    <!--pytest.mark.skip-->
    ```py
    all([
        # strictly decreasing sequences stay strictly decreasing
        all(X[1:] < X[:-1]) == all(C["$X"][1:] < C["$X"][:-1]),
        # strictly increasing sequences stay strictly increasing
        all(X[1:] > X[:-1]) == all(C["$X"][1:] > C["$X"][:-1]),
    ])
    ```
    ///

    The shape of the data neighbourhood is specified as an ordered list of
    unique data axes and boundary conditions that are applied to these axes.
    If the safeguard is applied to data with an insufficient number of
    dimensions, it raises an exception. If the safeguard is applied to data
    with additional dimensions, it is indendently applied along these extra
    axes. For instance, a 2d QoI is applied to independently to all 2d slices
    in a 3d data cube.

    If the data neighbourhood uses the [valid][....BoundaryCondition.valid]
    boundary condition along an axis, the safeguard is only applied to data
    neighbourhoods centred on data points that have sufficient points before
    and after to satisfy the neighbourhood shape, i.e. it is not applied to
    all data points. If the axis is smaller than required by the neighbourhood
    along this axis, the safeguard is not applied. Using a different
    [`BoundaryCondition`][....BoundaryCondition] ensures that the safeguard is
    always applied to all data points.

    If the derived quantity of interest for a data neighbourhood evaluates to
    an infinite value, this safeguard guarantees that the quantity of interest
    on the corrected data neighbourhood produces the exact same infinite value.
    For a NaN quantity of interest, this safeguard guarantees that the quantity
    of interest on the corrected data neighbourhood is also NaN, but does not
    guarantee that it has the same bit pattern.

    The error bound can be verified by evaluating the QoI in the floating-point
    data type selected by `qoi_dtype` parameter using the
    [`evaluate_qoi`][.evaluate_qoi] method.

    Please refer to the
    [`StencilQuantityOfInterestExpression`][.....qois.StencilQuantityOfInterestExpression]
    for the EBNF grammar that specifies the language in which the quantities of
    interest are written.

    The implementation was originally inspired by:

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
    neighbourhood : Sequence[dict[str, JSON] | NeighbourhoodBoundaryAxis]
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

        The error bound is applied relative to the values of the quantity of
        interest evaluated on the original data.

        If `eb` is a late-bound parameter, its late-bound value must be
        broadcastable to the shape of the data to be safeguarded, not the shape
        of the QoI that has been evaluated (even though it is only applied to
        the QoI). The [`expand_qoi_to_data_shape`][.expand_qoi_to_data_shape]
        method can be used to expand an error-bound from the QoI shape to the
        data shape.
    qoi_dtype : str | ToFloatMode
        The floating-point data type in which the quantity of interest is
        evaluated. By default, the smallest floating-point data type that can
        losslessly represent all input data values is chosen.

    Raises
    ------
    TypeCheckError
        if any parameter has the wrong type.
    SyntaxError
        if the `qoi` is not a valid stencil quantity of interest expression.
    ValueError
        if the `neighbourhood` is empty.
    ValueError
        if any `neighbourhood.axis` is not unique.
    ValueError
        if `type` does not name a valid error bound, or the `qoi_dtype` does
        not name a valid floating-point data type.
    ValueError
        if `eb` is an invalid error bound value for the error bound `type`.
    ...
        if instantiating a neighbourhood boundary axis raises an exception.
    """

    __slots__: tuple[str, ...] = (
        "_qoi",
        "_neighbourhood",
        "_type",
        "_eb",
        "_qoi_dtype",
        "_qoi_expr",
    )
    _qoi: StencilQuantityOfInterestExpression
    _neighbourhood: tuple[NeighbourhoodBoundaryAxis, ...]
    _type: ErrorBound
    _eb: int | float | Parameter
    _qoi_dtype: ToFloatMode
    _qoi_expr: StencilQuantityOfInterest

    kind: ClassVar[str] = "qoi_eb_stencil"

    def __init__(
        self,
        qoi: StencilQuantityOfInterestExpression,
        neighbourhood: Sequence[dict[str, JSON] | NeighbourhoodBoundaryAxis],
        type: str | ErrorBound,
        eb: int | float | str | Parameter,
        qoi_dtype: str | ToFloatMode = ToFloatMode.lossless,
    ) -> None:
        with ctx.safeguard(self):
            with ctx.parameter("qoi"):
                TypeCheckError.check_instance_or_raise(qoi, str)

            with ctx.parameter("neighbourhood"):
                TypeCheckError.check_instance_or_raise(neighbourhood, Sequence)

                all_axes: list[int] = []
                neighbourhood_: list[NeighbourhoodBoundaryAxis] = []
                for i, axis in enumerate(neighbourhood):
                    with ctx.index(i):
                        TypeCheckError.check_instance_or_raise(
                            axis, dict | NeighbourhoodBoundaryAxis
                        )
                        axis_ = (
                            axis
                            if isinstance(axis, NeighbourhoodBoundaryAxis)
                            else NeighbourhoodBoundaryAxis.from_config(axis)
                        )
                        if axis_.axis in all_axes:
                            raise ValueError("axis must be unique") | ctx
                        all_axes.append(axis_.axis)
                        neighbourhood_.append(axis_)
                self._neighbourhood = tuple(neighbourhood_)

                if len(self._neighbourhood) <= 0:
                    raise ValueError("must not be empty") | ctx

            with ctx.parameter("type"):
                TypeCheckError.check_instance_or_raise(type, str | ErrorBound)
                self._type = (
                    type
                    if isinstance(type, ErrorBound)
                    else lookup_enum_or_raise(ErrorBound, type)
                )

            with ctx.parameter("eb"):
                TypeCheckError.check_instance_or_raise(
                    eb, int | float | str | Parameter
                )
                if isinstance(eb, Parameter):
                    self._eb = eb
                elif isinstance(eb, str):
                    self._eb = Parameter(eb)
                else:
                    _check_error_bound(self._type, eb)
                    self._eb = eb

            with ctx.parameter("qoi_dtype"):
                TypeCheckError.check_instance_or_raise(qoi_dtype, str | ToFloatMode)
                self._qoi_dtype = (
                    qoi_dtype
                    if isinstance(qoi_dtype, ToFloatMode)
                    else lookup_enum_or_raise(ToFloatMode, qoi_dtype)
                )

            shape = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)
            I = tuple(axis.before for axis in self._neighbourhood)  # noqa: E741

            with ctx.parameter("qoi"):
                self._qoi = qoi
                self._qoi_expr = StencilQuantityOfInterest(
                    qoi, stencil_shape=shape, stencil_I=I
                )

    @property
    @override
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

    @override
    def compute_check_neighbourhood_for_data_shape(
        self,
        data_shape: tuple[int, ...],
    ) -> tuple[dict[BoundaryCondition, NeighbourhoodAxis], ...]:
        """
        Compute the shape of the data neighbourhood for data of a given shape.
        Boundary conditions of the same kind are combined, but separate kinds
        are tracked separately.

        An empty [`dict`][dict] is returned along dimensions for which the
        stencil QoI safeguard does not need to look at adjacent data points.

        This method also checks that the data shape is compatible with the
        stencil QoI safeguard.

        Parameters
        ----------
        data_shape : tuple[int, ...]
            The shape of the data.

        Returns
        -------
        neighbourhood_shape : tuple[dict[BoundaryCondition, NeighbourhoodAxis], ...]
            The shape of the data neighbourhood.

        Raises
        ------
        IndexError
            if any `neighbourhood` axis is out of bounds in `data_shape`.
        IndexError
            if any `neighbourhood` axis is duplicate.
        """

        neighbourhood: list[dict[BoundaryCondition, NeighbourhoodAxis]] = [
            dict() for _ in data_shape
        ]

        all_axes: list[int] = []
        with ctx.safeguard(self), ctx.parameter("neighbourhood"):
            for i, axis in enumerate(self._neighbourhood):
                with ctx.index(i), ctx.parameter("axis"):
                    if (axis.axis >= len(data_shape)) or (axis.axis < -len(data_shape)):
                        raise (
                            IndexError(
                                f"{axis.axis} is out of bounds for array of shape {data_shape}"
                            )
                            | ctx
                        )
                    naxis = axis.axis if axis.axis >= 0 else len(data_shape) + axis.axis
                    if naxis in all_axes:
                        raise (
                            IndexError(
                                f"duplicate axis index {axis.axis}, normalised to {naxis}, for array of shape {data_shape}"
                            )
                            | ctx
                        )
                    all_axes.append(naxis)

                    neighbourhood[naxis][axis.boundary] = axis.shape

        if any(a == 0 for a in data_shape):
            return tuple(dict() for _ in data_shape)

        return tuple(neighbourhood)

    def evaluate_qoi(
        self,
        data: np.ndarray[S, np.dtype[T]],
        late_bound: Bindings,
    ) -> np.ndarray[tuple[int, ...], np.dtype[F]]:
        """
        Evaluate the derived quantity of interest on the `data` in the
        floating-point data type selected by the `qoi_dtype` parameter.

        The quantity of interest may have a different shape if the
        [valid][.....BoundaryCondition.valid] boundary condition is used along
        any axis.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Data for which the quantity of interest is evaluated.
        late_bound : Bindings
            Bindings for late-bound constants in the quantity of interest.

        Returns
        -------
        qoi : np.ndarray[tuple[int, ...], np.dtype[F]]
            Evaluated quantity of interest, in floating-point.

        Raises
        ------
        IndexError
            if any `neighbourhood` axis is out of bounds in `data`.
        IndexError
            if any `neighbourhood` axis is duplicate.
        TypeError
            if the `data` could not be losslessly cast to `qoi_dtype`.
        LateBoundParameterResolutionError
            if any `neighbourhood` `axis.constant_boundary` is late-bound but
            its late-bound parameter is not in `late_bound`.
        ValueError
            if any `neighbourhood` `axis.constant_boundary` is late-bound but
            not a scalar.
        TypeError
            if any `neighbourhood` `axis.constant_boundary` is floating-point
            but the `data` is integer.
        ValueError
            if any `neighbourhood` `axis.constant_boundary` could not be
            losslessly converted to the `data`'s type.
        LateBoundParameterResolutionError
            if any of the `qoi`'s late-bound constants is not contained in the
            bindings.
        ValueError
            if any late-bound constant could not be broadcast to the `data`'s
            shape.
        TypeError
            if any late-bound constant is floating-point but the `data` is
            integer.
        ValueError
            if not all values for all late-bound constants could be losslessly
            converted to the `data`'s type.
        """

        # check that the data shape is compatible with the neighbourhood shape
        self.compute_check_neighbourhood_for_data_shape(data.shape)

        empty_shape = list(data.shape)

        # if the neighbourhood is empty, e.g. because we are in valid mode and
        #  the neighbourhood shape exceeds the data shape, return empty
        for axis in self._neighbourhood:
            if axis.boundary == BoundaryCondition.valid:
                empty_shape[axis.axis] = max(
                    0, empty_shape[axis.axis] - axis.before - axis.after
                )

        with ctx.safeguard(self):
            with ctx.parameter("qoi_dtype"):
                ftype: np.dtype[F] = self._qoi_dtype.floating_point_dtype_for(
                    data.dtype
                )  # type: ignore

            if any(s == 0 for s in empty_shape):
                return _zeros(tuple(empty_shape), dtype=ftype)

            constant_boundaries: list[None | np.ndarray[tuple[()], np.dtype[T]]] = []
            with ctx.parameter("neighbourhood"):
                for i, axis in enumerate(self._neighbourhood):
                    with ctx.index(i), ctx.parameter("constant_boundary"):
                        constant_boundaries.append(
                            None
                            if axis.constant_boundary is None
                            else late_bound.resolve_ndarray_with_lossless_cast(
                                axis.constant_boundary, (), data.dtype
                            )
                            if isinstance(axis.constant_boundary, Parameter)
                            else lossless_cast(axis.constant_boundary, data.dtype)
                        )

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

            data_windows_float_: np.ndarray[tuple[int, ...], np.dtype[F]] = to_float(
                _sliding_window_view(
                    data_boundary,
                    window,
                    axis=tuple(axis.axis for axis in self._neighbourhood),
                    writeable=False,
                ),
                ftype=ftype,
            )
            qoi_shape: tuple[int, ...] = data_windows_float_.shape[: -len(window)]
            data_windows_float: np.ndarray[
                tuple[int, *tuple[int, ...]], np.dtype[F]
            ] = _reshape(data_windows_float_, (-1, *window))

            late_bound_constants: dict[
                Parameter, np.ndarray[tuple[int, *tuple[int, ...]], np.dtype[F]]
            ] = dict()
            with ctx.parameter("qoi"):
                for c in self._qoi_expr.late_bound_constants:
                    with ctx.parameter(c):
                        late_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = (
                            late_bound.resolve_ndarray_with_lossless_cast(
                                c, data.shape, data.dtype
                            )
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
                    late_windows_float_: np.ndarray[tuple[int, ...], np.dtype[F]] = (
                        to_float(
                            _sliding_window_view(
                                late_boundary,
                                window,
                                axis=tuple(axis.axis for axis in self._neighbourhood),
                                writeable=False,
                            ),
                            ftype=ftype,
                        )
                    )
                    late_windows_float: np.ndarray[
                        tuple[int, *tuple[int, ...]], np.dtype[F]
                    ] = _reshape(late_windows_float_, (-1, *window))
                    late_bound_constants[c] = late_windows_float

        qoi_: np.ndarray[tuple[int], np.dtype[F]] = self._qoi_expr.eval(
            data_windows_float,
            late_bound_constants,
        )
        qoi: np.ndarray[tuple[int, ...], np.dtype[F]] = qoi_.reshape(qoi_shape)
        return qoi

    @override
    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements in the `prediction` array satisfy the error bound
        for the quantity of interest over a neighbourhood on the `data`.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Original data array, relative to which the `prediction` is checked.
        prediction : np.ndarray[S, np.dtype[T]]
            Prediction for the `data` array.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only check at data points where the condition is [`True`][True].

        Returns
        -------
        ok : np.ndarray[S, np.dtype[np.bool]]
            Pointwise, `True` if the check succeeded for this element.

        Raises
        ------
        IndexError
            if any `neighbourhood` axis is out of bounds in `data`.
        IndexError
            if any `neighbourhood` axis is duplicate.
        TypeError
            if the `data` could not be losslessly cast to `qoi_dtype`.
        LateBoundParameterResolutionError
            if any `neighbourhood` `axis.constant_boundary` is late-bound but
            its late-bound parameter is not in `late_bound`.
        ValueError
            if any `neighbourhood` `axis.constant_boundary` is late-bound but
            not a scalar.
        TypeError
            if any `neighbourhood` `axis.constant_boundary` is floating-point
            but the `data` is integer.
        ValueError
            if any `neighbourhood` `axis.constant_boundary` could not be
            losslessly converted to the `data`'s type.
        LateBoundParameterResolutionError
            if any of the `qoi`'s late-bound constants is not contained in the
            bindings.
        ValueError
            if any late-bound constant could not be broadcast to the `data`'s
            shape.
        TypeError
            if any late-bound constant is floating-point but the `data` is
            integer.
        ValueError
            if not all values for all late-bound constants could be losslessly
            converted to the `data`'s type.
        LateBoundParameterResolutionError
            if the error bound `eb` is late-bound but its late-bound parameter
            is not in `late_bound`.
        ValueError
            if the late-bound `eb` could not be broadcast to the `data`'s
            shape.
        ValueError
            if the late-bound `eb` is non-finite, i.e. infinite or NaN, or an
            invalid error bound value for the error bound `type`.
        """

        # check that the data shape is compatible with the neighbourhood shape
        self.compute_check_neighbourhood_for_data_shape(data.shape)

        window = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)

        # if the neighbourhood is empty, e.g. because we are in valid mode and
        #  the neighbourhood shape exceeds the data shape, allow all values
        for axis, w in zip(self._neighbourhood, window):
            if data.shape[axis.axis] < w and axis.boundary == BoundaryCondition.valid:
                return _ones(data.shape, dtype=np.dtype(np.bool))
        if data.size == 0:
            return _ones(data.shape, dtype=np.dtype(np.bool))

        with ctx.safeguard(self):
            constant_boundaries: list[None | np.ndarray[tuple[()], np.dtype[T]]] = []
            with ctx.parameter("neighbourhood"):
                for i, axis in enumerate(self._neighbourhood):
                    with ctx.index(i), ctx.parameter("constant_boundary"):
                        constant_boundaries.append(
                            None
                            if axis.constant_boundary is None
                            else late_bound.resolve_ndarray_with_lossless_cast(
                                axis.constant_boundary, (), data.dtype
                            )
                            if isinstance(axis.constant_boundary, Parameter)
                            else lossless_cast(axis.constant_boundary, data.dtype)
                        )

            data_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = data
            prediction_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = prediction
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
                prediction_boundary = _pad_with_boundary(
                    prediction_boundary,
                    axis.boundary,
                    axis.before,
                    axis.after,
                    axis_constant_boundary,
                    axis.axis,
                )

            with ctx.parameter("qoi_dtype"):
                ftype: np.dtype[np.floating] = self._qoi_dtype.floating_point_dtype_for(
                    data.dtype
                )

            data_windows_float_: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
                to_float(
                    _sliding_window_view(
                        data_boundary,
                        window,
                        axis=tuple(axis.axis for axis in self._neighbourhood),
                        writeable=False,
                    ),
                    ftype=ftype,
                )
            )
            qoi_shape: tuple[int, ...] = data_windows_float_.shape[: -len(window)]
            prediction_windows_float_: np.ndarray[
                tuple[int, ...], np.dtype[np.floating]
            ] = to_float(
                _sliding_window_view(
                    prediction_boundary,
                    window,
                    axis=tuple(axis.axis for axis in self._neighbourhood),
                    writeable=False,
                ),
                ftype=ftype,
            )

            data_windows_float: np.ndarray[
                tuple[int, *tuple[int, ...]], np.dtype[np.floating]
            ] = _reshape(data_windows_float_, (-1, *window))
            prediction_windows_float: np.ndarray[
                tuple[int, *tuple[int, ...]], np.dtype[np.floating]
            ] = _reshape(prediction_windows_float_, (-1, *window))

            late_bound_constants: dict[
                Parameter,
                np.ndarray[tuple[int, *tuple[int, ...]], np.dtype[np.floating]],
            ] = dict()
            with ctx.parameter("qoi"):
                for c in self._qoi_expr.late_bound_constants:
                    with ctx.parameter(c):
                        late_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = (
                            late_bound.resolve_ndarray_with_lossless_cast(
                                c, data.shape, data.dtype
                            )
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
                    late_windows_float_: np.ndarray[
                        tuple[int, ...], np.dtype[np.floating]
                    ] = to_float(
                        _sliding_window_view(
                            late_boundary,
                            window,
                            axis=tuple(axis.axis for axis in self._neighbourhood),
                            writeable=False,
                        ),
                        ftype=ftype,
                    )
                    late_windows_float: np.ndarray[
                        tuple[int, *tuple[int, ...]], np.dtype[np.floating]
                    ] = _reshape(late_windows_float_, (-1, *window))
                    late_bound_constants[c] = late_windows_float

            valid_slice = [slice(None)] * data.ndim
            for axis in self._neighbourhood:
                if axis.boundary == BoundaryCondition.valid:
                    start = None if axis.before == 0 else axis.before
                    end = None if axis.after == 0 else -axis.after
                    valid_slice[axis.axis] = slice(start, end)

            # optimization: only evaluate the QoI where necessary
            if where is not True:
                where_flat = where[tuple(valid_slice)].flatten()
                data_windows_float = np.compress(where_flat, data_windows_float, axis=0)
                prediction_windows_float = np.compress(
                    where_flat, prediction_windows_float, axis=0
                )
                late_bound_constants = {
                    c: np.compress(where_flat, cv, axis=0)
                    for (c, cv) in late_bound_constants.items()
                }

            qoi_data: np.ndarray[tuple[int], np.dtype[np.floating]] = (
                self._qoi_expr.eval(
                    data_windows_float,
                    late_bound_constants,
                )
            )
            qoi_prediction: np.ndarray[tuple[int], np.dtype[np.floating]] = (
                self._qoi_expr.eval(
                    prediction_windows_float,
                    late_bound_constants,
                )
            )

            with ctx.parameter("eb"):
                eb_: np.ndarray[tuple[()] | tuple[int, ...], np.dtype[np.floating]] = (
                    late_bound.resolve_ndarray_with_saturating_finite_float_cast(
                        self._eb,
                        data.shape,  # for simplicity, we resolve to the data shape
                        ftype,
                    )
                    if isinstance(self._eb, Parameter)
                    else saturating_finite_float_cast(self._eb, ftype)
                )
                if isinstance(self._eb, Parameter):
                    _eb: Parameter = self._eb
                    with ctx.late_bound_parameter(_eb):
                        # and then truncate it
                        eb_ = self.truncate_data_to_qoi_shape(eb_)
                        _check_error_bound(self._type, eb_)
                eb: np.ndarray[tuple[()] | tuple[int], np.dtype[np.floating]] = (
                    eb_ if eb_.shape == () else eb_.flatten()  # type: ignore
                )
                if where is not True and eb.shape != ():
                    eb = np.compress(where_flat, eb, axis=0)

        finite_ok: np.ndarray[tuple[int], np.dtype[np.bool]] = np.less_equal(
            _compute_finite_absolute_error(self._type, qoi_data, qoi_prediction),
            _compute_finite_absolute_error_bound(self._type, eb, qoi_data),
        )

        windows_ok_: np.ndarray[tuple[int], np.dtype[np.bool]] = _ensure_array(
            finite_ok
        )
        np.equal(qoi_data, qoi_prediction, out=windows_ok_, where=np.isinf(qoi_data))
        np.isnan(qoi_prediction, out=windows_ok_, where=np.isnan(qoi_data))

        # the check succeeds where `where` is False
        windows_ok: np.ndarray[tuple[int], np.dtype[np.bool]]
        if where is True:
            windows_ok = windows_ok_
        else:
            windows_ok = _ones(where_flat.shape, np.dtype(np.bool))
            _place(windows_ok, where_flat, windows_ok_)

        # the check succeeds for boundary points that were excluded by a valid
        #  boundary
        ok: np.ndarray[S, np.dtype[np.bool]] = _ones(
            data.shape, dtype=np.dtype(np.bool)
        )
        ok[tuple(valid_slice)] = windows_ok.reshape(qoi_shape)

        return ok

    @override
    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
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
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the safe intervals at pointwise checks where the
            condition is [`True`][True].

        Returns
        -------
        intervals : IntervalUnion[T, int, int]
            Union of intervals in which the error bound is upheld.

        Raises
        ------
        IndexError
            if any `neighbourhood` axis is out of bounds in `data`.
        IndexError
            if any `neighbourhood` axis is duplicate.
        TypeError
            if the `data` could not be losslessly cast to `qoi_dtype`.
        LateBoundParameterResolutionError
            if any `neighbourhood` `axis.constant_boundary` is late-bound but
            its late-bound parameter is not in `late_bound`.
        ValueError
            if any `neighbourhood` `axis.constant_boundary` is late-bound but
            not a scalar.
        TypeError
            if any `neighbourhood` `axis.constant_boundary` is floating-point
            but the `data` is integer.
        ValueError
            if any `neighbourhood` `axis.constant_boundary` could not be
            losslessly converted to the `data`'s type.
        LateBoundParameterResolutionError
            if any of the `qoi`'s late-bound constants is not contained in the
            bindings.
        ValueError
            if any late-bound constant could not be broadcast to the `data`'s
            shape.
        TypeError
            if any late-bound constant is floating-point but the `data` is
            integer.
        ValueError
            if not all values for all late-bound constants could be losslessly
            converted to the `data`'s type.
        LateBoundParameterResolutionError
            if the error bound `eb` is late-bound but its late-bound parameter
            is not in `late_bound`.
        ValueError
            if the late-bound `eb` could not be broadcast to the `data`'s
            shape.
        ValueError
            if the late-bound `eb` is non-finite, i.e. infinite or NaN, or an
            invalid error bound value for the error bound `type`.
        """

        # check that the data shape is compatible with the neighbourhood shape
        self.compute_check_neighbourhood_for_data_shape(data.shape)

        window = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)

        # if the neighbourhood is empty, e.g. because we are in valid mode and
        #  the neighbourhood shape exceeds the data shape, allow all values
        for axis, w in zip(self._neighbourhood, window):
            if data.shape[axis.axis] < w and axis.boundary == BoundaryCondition.valid:
                return Interval.full_like(data).into_union()
        if data.size == 0:
            return Interval.full_like(data).into_union()

        with ctx.safeguard(self):
            constant_boundaries: list[None | np.ndarray[tuple[()], np.dtype[T]]] = []
            with ctx.parameter("neighbourhood"):
                for i, axis in enumerate(self._neighbourhood):
                    with ctx.index(i), ctx.parameter("constant_boundary"):
                        constant_boundaries.append(
                            None
                            if axis.constant_boundary is None
                            else late_bound.resolve_ndarray_with_lossless_cast(
                                axis.constant_boundary, (), data.dtype
                            )
                            if isinstance(axis.constant_boundary, Parameter)
                            else lossless_cast(axis.constant_boundary, data.dtype)
                        )

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

            with ctx.parameter("qoi_dtype"):
                ftype: np.dtype[np.floating] = self._qoi_dtype.floating_point_dtype_for(
                    data.dtype
                )

            data_windows_float_: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
                to_float(
                    _sliding_window_view(
                        data_boundary,
                        window,
                        axis=tuple(axis.axis for axis in self._neighbourhood),
                        writeable=False,
                    ),
                    ftype=ftype,
                )
            )
            qoi_stencil_shape: tuple[int, ...] = data_windows_float_.shape
            data_windows_float: np.ndarray[
                tuple[int, *tuple[int, ...]], np.dtype[np.floating]
            ] = _reshape(data_windows_float_, (-1, *window))

            late_bound_constants: dict[
                Parameter,
                np.ndarray[tuple[int, *tuple[int, ...]], np.dtype[np.floating]],
            ] = dict()
            with ctx.parameter("qoi"):
                for c in self._qoi_expr.late_bound_constants:
                    with ctx.parameter(c):
                        late_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = (
                            late_bound.resolve_ndarray_with_lossless_cast(
                                c, data.shape, data.dtype
                            )
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
                    late_windows_float_: np.ndarray[
                        tuple[int, ...], np.dtype[np.floating]
                    ] = to_float(
                        _sliding_window_view(
                            late_boundary,
                            window,
                            axis=tuple(axis.axis for axis in self._neighbourhood),
                            writeable=False,
                        ),
                        ftype=ftype,
                    )
                    late_windows_float: np.ndarray[
                        tuple[int, *tuple[int, ...]], np.dtype[np.floating]
                    ] = _reshape(late_windows_float_, (-1, *window))
                    late_bound_constants[c] = late_windows_float

            valid_slice = [slice(None)] * data.ndim
            for axis in self._neighbourhood:
                if axis.boundary == BoundaryCondition.valid:
                    start = None if axis.before == 0 else axis.before
                    end = None if axis.after == 0 else -axis.after
                    valid_slice[axis.axis] = slice(start, end)

            # optimization: only evaluate the QoI and compute data bounds where
            #  necessary
            if where is not True:
                where_flat = where[tuple(valid_slice)].flatten()
                data_windows_float = np.compress(where_flat, data_windows_float, axis=0)
                late_bound_constants = {
                    c: np.compress(where_flat, cv, axis=0)
                    for (c, cv) in late_bound_constants.items()
                }

            data_qoi: np.ndarray[tuple[int], np.dtype[np.floating]] = (
                self._qoi_expr.eval(
                    data_windows_float,
                    late_bound_constants,
                )
            )

            with ctx.parameter("eb"):
                eb_: np.ndarray[tuple[()] | tuple[int, ...], np.dtype[np.floating]] = (
                    late_bound.resolve_ndarray_with_saturating_finite_float_cast(
                        self._eb,
                        data.shape,  # for simplicity, we resolve to the data shape
                        ftype,
                    )
                    if isinstance(self._eb, Parameter)
                    else saturating_finite_float_cast(self._eb, ftype)
                )
                if isinstance(self._eb, Parameter):
                    _eb: Parameter = self._eb
                    with ctx.late_bound_parameter(_eb):
                        eb_ = self.truncate_data_to_qoi_shape(
                            eb_
                        )  # and then truncate it
                        _check_error_bound(self._type, eb_)
                eb: np.ndarray[tuple[()] | tuple[int], np.dtype[np.floating]] = (
                    eb_ if eb_.shape == () else eb_.flatten()  # type: ignore
                )
                if where is not True and eb.shape != ():
                    eb = np.compress(where_flat, eb, axis=0)

        qoi_lower_upper: tuple[
            np.ndarray[tuple[int], np.dtype[np.floating]],
            np.ndarray[tuple[int], np.dtype[np.floating]],
        ] = _apply_finite_qoi_error_bound(
            self._type,
            eb,
            data_qoi,
        )
        qoi_lower, qoi_upper = qoi_lower_upper

        # compute the bounds in data space
        data_windows_float_lower_, data_windows_float_upper_ = (
            self._qoi_expr.compute_data_bounds(
                qoi_lower,
                qoi_upper,
                data_windows_float,
                late_bound_constants,
            )
        )

        # the data bounds can be arbitrary where `where` is False since they
        #  are later overridden by preserve_only_where
        data_windows_float_lower: np.ndarray[tuple[int, ...], np.dtype[np.floating]]
        data_windows_float_upper: np.ndarray[tuple[int, ...], np.dtype[np.floating]]
        if where is True:
            data_windows_float_lower = data_windows_float_lower_.reshape(
                qoi_stencil_shape
            )
            data_windows_float_upper = data_windows_float_upper_.reshape(
                qoi_stencil_shape
            )
        else:
            where_indices = np.nonzero(where_flat)[0].reshape(
                (-1, *tuple(1 for _ in window))
            )
            data_windows_float_lower = np.full(
                qoi_stencil_shape, -np.inf, dtype=ftype
            ).reshape(-1, *window)
            data_windows_float_upper = np.full(
                qoi_stencil_shape, np.inf, dtype=ftype
            ).reshape(-1, *window)
            np.put_along_axis(
                data_windows_float_lower,
                where_indices,
                data_windows_float_lower_,
                axis=0,
            )
            np.put_along_axis(
                data_windows_float_upper,
                where_indices,
                data_windows_float_upper_,
                axis=0,
            )
            data_windows_float_lower = data_windows_float_lower.reshape(
                qoi_stencil_shape
            )
            data_windows_float_upper = data_windows_float_upper.reshape(
                qoi_stencil_shape
            )

        # only contribute window elements that are used in the QoI
        window_used = _zeros(window, dtype=np.dtype(np.bool))
        for idxs in self._qoi_expr.data_indices:
            window_used[idxs] = True

        # compute the reverse: for each data element, which windows is it in
        # i.e. for each data element, which QoI elements does it contribute to
        #      and thus which data bounds affect it
        reverse_indices_windows = _reverse_neighbourhood_indices(
            data.shape,
            self._neighbourhood,
            window_used,
            True if where is True else where_flat,
        )

        # flatten the QoI data bounds and append an infinite value,
        #  which is indexed if an element did not contribute to the maximum
        #  number of windows
        data_windows_float_lower_flat = np.full(
            data_windows_float_lower.size + 1, -np.inf, dtype=ftype
        )
        data_windows_float_upper_flat = np.full(
            data_windows_float_upper.size + 1, np.inf, dtype=ftype
        )
        data_windows_float_lower_flat[:-1] = data_windows_float_lower.flatten()
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

        # only preserve intervals where some (neighbouring) data point
        #  contributed (back) safety requirements
        # this takes the `where` condition, after overlapping stencils, into
        #  account
        return compute_safe_data_lower_upper_interval_union(
            data, data_float_lower, data_float_upper
        ).preserve_only_where(
            (  # type: ignore
                reverse_indices_windows[:, 0] < data_windows_float_lower.size
            )
            # where `where` is not False, at least the isnan stays isnan rule
            #  should always be honoured, anything else would be unexpected
            | (True if where is True else where.flatten())
        )

    @override
    def compute_footprint(
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Compute the footprint of the `foot` array, e.g. for expanding data
        points into the pointwise checks that they contribute to.

        The footprint usually extends beyond `foot & where`.

        Parameters
        ----------
        foot : np.ndarray[S, np.dtype[np.bool]]
            Array for which the footprint is computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the footprint at pointwise checks where the condition
            is [`True`][True].

            Conceptually, `where` is applied to `footprint` at the end.

        Returns
        -------
        print : np.ndarray[S, np.dtype[np.bool]]
            The footprint of the `foot` array.
        """

        # check that the foot shape is compatible with the neighbourhood shape
        self.compute_check_neighbourhood_for_data_shape(foot.shape)

        window = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)

        # if the neighbourhood is empty, e.g. because we are in valid mode and
        #  the neighbourhood shape exceeds the data shape, the footprint is
        #  empty
        for axis, w in zip(self._neighbourhood, window):
            if foot.shape[axis.axis] < w and axis.boundary == BoundaryCondition.valid:
                return _zeros(foot.shape, dtype=np.dtype(np.bool))
        if foot.size == 0:
            return _zeros(foot.shape, dtype=np.dtype(np.bool))

        foot_boundary: np.ndarray[tuple[int, ...], np.dtype[np.bool]] = foot
        for axis in self._neighbourhood:
            foot_boundary = _pad_with_boundary(
                foot_boundary,
                axis.boundary,
                axis.before,
                axis.after,
                None
                if axis.constant_boundary is None
                else _zeros((), np.dtype(np.bool)),
                axis.axis,
            )

        foot_windows: np.ndarray[tuple[int, ...], np.dtype[np.bool]] = (
            _sliding_window_view(
                foot_boundary,
                window,
                axis=tuple(axis.axis for axis in self._neighbourhood),
                writeable=False,
            )
        )

        valid_slice = [slice(None)] * foot.ndim
        for axis in self._neighbourhood:
            if axis.boundary == BoundaryCondition.valid:
                start = None if axis.before == 0 else axis.before
                end = None if axis.after == 0 else -axis.after
                valid_slice[axis.axis] = slice(start, end)

        # only contribute window elements that are used in the QoI
        window_used = _zeros(window, dtype=np.dtype(np.bool))
        for idxs in self._qoi_expr.data_indices:
            window_used[idxs] = True

        # expand into the footprint
        footprint = np.zeros_like(foot)
        footprint[tuple(valid_slice)] = np.logical_or.reduce(
            (foot_windows & window_used.reshape(((1,) * foot.ndim) + window)).reshape(
                foot_windows.shape[: foot.ndim] + (-1,)
            ),
            axis=foot.ndim,
        )
        footprint &= where

        return footprint

    @override
    def compute_inverse_footprint(
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Compute the inverse footprint of the `foot` array, e.g. for expanding
        pointwise check fails into the points that could have contributed to
        the failures.

        The inverse footprint usually extends beyond `foot & where`.

        Parameters
        ----------
        foot : np.ndarray[S, np.dtype[np.bool]]
            Array for which the inverse footprint is computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the inverse footprint at pointwise checks where the
            condition is [`True`][True].

            Conceptually, `where` is applied to `foot` at the start.

        Returns
        -------
        print : np.ndarray[S, np.dtype[np.bool]]
            The inverse footprint of the `foot` array.
        """

        # check that the foot shape is compatible with the neighbourhood shape
        self.compute_check_neighbourhood_for_data_shape(foot.shape)

        window = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)

        # if the neighbourhood is empty, e.g. because we are in valid mode and
        #  the neighbourhood shape exceeds the data shape, the inverse
        #  footprint is empty
        for axis, w in zip(self._neighbourhood, window):
            if foot.shape[axis.axis] < w and axis.boundary == BoundaryCondition.valid:
                return _zeros(foot.shape, dtype=np.dtype(np.bool))
        if foot.size == 0:
            return _zeros(foot.shape, dtype=np.dtype(np.bool))

        valid_slice = [slice(None)] * foot.ndim
        for axis in self._neighbourhood:
            if axis.boundary == BoundaryCondition.valid:
                start = None if axis.before == 0 else axis.before
                end = None if axis.after == 0 else -axis.after
                valid_slice[axis.axis] = slice(start, end)

        # only contribute window elements that are used in the QoI
        window_used = _zeros(window, dtype=np.dtype(np.bool))
        for idxs in self._qoi_expr.data_indices:
            window_used[idxs] = True

        # valid foot, which includes the where condition
        foot_valid = (foot & where)[tuple(valid_slice)]

        # compute the reverse: for each data element, which windows is it in
        # i.e. for each data element, which QoI elements does it contribute to
        #      and thus which foot elements affect it
        reverse_indices_windows = _reverse_neighbourhood_indices(
            foot.shape,
            self._neighbourhood,
            window_used,
            # optimization: only compute the reverse indices where the foot is
            #  True, since the others don't contribute anyways
            where_flat=foot_valid.flatten(),
        )

        window_size = reduce(lambda x, y: x * y, window, 1)

        # optimization: since we only compute reverse indices where the foot is
        #  True, we can more efficiently check which elements are part of the
        #  inverse footprint by checking if they have any valid reverse
        #  contribution indices
        valid_reverse_index = reverse_indices_windows != (foot_valid.size * window_size)

        inverse_footprint: np.ndarray[S, np.dtype[np.bool]] = np.logical_or.reduce(
            valid_reverse_index, axis=1
        ).reshape(foot.shape)

        return inverse_footprint

    def truncate_data_to_qoi_shape(
        self, data: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[T]]:
        """
        Truncate the `data` array to the shape of the `qoi` array that would be evaluated from it.

        For neighbourhoods without any [valid][.....BoundaryCondition.valid]
        boundaries, this method simply copies the `data` array. Along axes with
        a [valid][.....BoundaryCondition.valid] boundary, the `data` is
        truncated before and after by as many elements as the boundary cuts off.

        This method is the *lossy* inverse of
        [`expand_qoi_to_data_shape`][..expand_qoi_to_data_shape].

        Parameters
        ----------
        data : np.ndarray[tuple[int, ...], np.dtype[T]]
            Data array to be truncated.

        Returns
        -------
        data_truncated : np.ndarray[tuple[int, ...], np.dtype[T]]
            Truncated data array.

        Raises
        ------
        IndexError
            if any `neighbourhood` axis is out of bounds in `data`.
        IndexError
            if any `neighbourhood` axis is duplicate.
        """

        data = _ensure_array(data)

        qoi_index = [slice(None, None) for _ in data.shape]

        all_axes: list[int] = []
        for axis in self._neighbourhood:
            if (axis.axis >= data.ndim) or (axis.axis < -data.ndim):
                raise (
                    IndexError(
                        f"axis index {axis.axis} is out of bounds for array of shape {data.shape}"
                    )
                    | ctx
                )
            naxis = axis.axis if axis.axis >= 0 else data.ndim + axis.axis
            if naxis in all_axes:
                raise (
                    IndexError(
                        f"duplicate axis index {axis.axis}, normalised to {naxis}, for array of shape {data.shape}"
                    )
                    | ctx
                )
            all_axes.append(naxis)

            if axis.boundary != BoundaryCondition.valid:
                continue

            qoi_index[naxis] = slice(
                axis.before, None if axis.after == 0 else -axis.after
            )

        return data[tuple(qoi_index)]

    def expand_qoi_to_data_shape(
        self, qoi: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[T]]:
        """
        Expand the `qoi` array to the shape of the data it was evaluated from.

        The `qoi` array is assumed to have been produced by the
        [`evaluate_qoi`][..evaluate_qoi] method.

        For neighbourhoods without any [valid][.....BoundaryCondition.valid]
        boundaries, this method simply copies the `qoi` array. Along axes with
        a [valid][.....BoundaryCondition.valid] boundary, the `qoi` is expanded
        before and after by as many elements as the boundary cut off. These new
        elements are filled with zeros.

        This method can be used to produce the array for the late-bound `eb`
        parameter by first deriving the error bound from an evaluated QoI array
        (shape) and then expanding it to the data shape.

        This method is the *lossy* inverse of
        [`truncate_data_to_qoi_shape`][..truncate_data_to_qoi_shape].

        Parameters
        ----------
        qoi : np.ndarray[tuple[int, ...], np.dtype[T]]
            QoI array to be expanded.

        Returns
        -------
        qoi_expanded : np.ndarray[tuple[int, ...], np.dtype[T]]
            Zero-expanded QoI array.

        Raises
        ------
        IndexError
            if any `neighbourhood` axis is out of bounds in `qoi`.
        IndexError
            if any `neighbourhood` axis is duplicate.
        """

        qoi = _ensure_array(qoi)

        data_shape = list(qoi.shape)
        qoi_index = [slice(None, None) for _ in qoi.shape]

        all_axes: list[int] = []
        for axis in self._neighbourhood:
            if (axis.axis >= qoi.ndim) or (axis.axis < -qoi.ndim):
                raise (
                    IndexError(
                        f"axis index {axis.axis} is out of bounds for array of shape {qoi.shape}"
                    )
                    | ctx
                )
            naxis = axis.axis if axis.axis >= 0 else qoi.ndim + axis.axis
            if naxis in all_axes:
                raise (
                    IndexError(
                        f"duplicate axis index {axis.axis}, normalised to {naxis}, for array of shape {qoi.shape}"
                    )
                    | ctx
                )
            all_axes.append(naxis)

            if axis.boundary != BoundaryCondition.valid:
                continue

            data_shape[naxis] += axis.before + axis.after
            qoi_index[naxis] = slice(
                axis.before, None if axis.after == 0 else -axis.after
            )

        out: np.ndarray[tuple[int, ...], np.dtype[T]] = _zeros(
            tuple(data_shape), dtype=qoi.dtype
        )
        out[tuple(qoi_index)] = qoi
        return out

    @override
    def get_config(self) -> dict[str, JSON]:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict[str, JSON]
            Configuration of the safeguard.
        """

        return dict(
            kind=type(self).kind,
            qoi=self._qoi,
            neighbourhood=[axis.get_config() for axis in self._neighbourhood],
            type=self._type.name,
            eb=self._eb,
            qoi_dtype=self._qoi_dtype.name,
        )
