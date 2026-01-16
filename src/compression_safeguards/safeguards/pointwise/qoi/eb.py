"""
Pointwise quantity of interest (QoI) error bound safeguard.
"""

__all__ = ["PointwiseQuantityOfInterestErrorBoundSafeguard"]

from collections.abc import Set
from typing import ClassVar, Literal

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import _ensure_array, _logical_and, _ones, _place, _reshape
from ....utils.bindings import Bindings, Parameter
from ....utils.cast import ToFloatMode, saturating_finite_float_cast, to_float
from ....utils.error import TypeCheckError, ctx, lookup_enum_or_raise
from ....utils.intervals import IntervalUnion
from ....utils.typing import JSON, F, S, T
from ..._qois import PointwiseQuantityOfInterest
from ..._qois.interval import compute_safe_data_lower_upper_interval_union
from ...eb import (
    ErrorBound,
    _apply_finite_qoi_error_bound,
    _check_error_bound,
    _compute_finite_absolute_error,
    _compute_finite_absolute_error_bound,
)
from ...qois import PointwiseQuantityOfInterestExpression
from ..abc import PointwiseSafeguard


class PointwiseQuantityOfInterestErrorBoundSafeguard(PointwiseSafeguard):
    """
    The `PointwiseQuantityOfInterestErrorBoundSafeguard` guarantees that the
    pointwise error `type` on a derived pointwise quantity of interest (QoI)
    is less than or equal to the provided bound `eb`.

    The quantity of interest is specified as a non-constant expression, in
    string form, over the pointwise value `x`. For example, to bound the error
    on the square of `x`, set `qoi="square(x)"` (or `qoi="x**2"`).

    If the derived quantity of interest for an element evaluates to an infinite
    value, this safeguard guarantees that the quantity of interest on the
    corrected value produces the exact same infinite value. For a NaN quantity
    of interest, this safeguard guarantees that the quantity of interest on the
    corrected value is also NaN, but does not guarantee that it has the same
    bit pattern.

    The error bound can be verified by evaluating the QoI in the floating-point
    data type selected by `qoi_dtype` parameter using the
    [`evaluate_qoi`][.evaluate_qoi] method.

    Please refer to the
    [`PointwiseQuantityOfInterestExpression`][.....qois.PointwiseQuantityOfInterestExpression]
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
    qoi : PointwiseExpr
        The non-constant expression for computing the derived quantity of
        interest over a pointwise value `x`.
    type : str | ErrorBound
        The type of error bound on the quantity of interest that is enforced by
        this safeguard.
    eb : int | float | str | Parameter
        The value of or late-bound parameter name for the error bound on the
        quantity of interest that is enforced by this safeguard.

        The error bound is applied relative to the values of the quantity of
        interest evaluated on the original data.
    qoi_dtype : str | ToFloatMode
        The floating-point data type in which the quantity of interest is
        evaluated. By default, the smallest floating-point data type that can
        losslessly represent all input data values is chosen.

    Raises
    ------
    TypeCheckError
        if any parameter has the wrong type.
    SyntaxError
        if the `qoi` is not a valid pointwise quantity of interest expression.
    ValueError
        if `type` does not name a valid error bound, or the `qoi_dtype` does
        not name a valid floating-point data type.
    ValueError
        if `eb` is an invalid error bound value for the error bound `type`.
    """

    __slots__: tuple[str, ...] = (
        "_qoi",
        "_type",
        "_eb",
        "_qoi_dtype",
        "_qoi_expr",
    )
    _qoi: PointwiseQuantityOfInterestExpression
    _type: ErrorBound
    _eb: int | float | Parameter
    _qoi_dtype: ToFloatMode
    _qoi_expr: PointwiseQuantityOfInterest

    kind: ClassVar[str] = "qoi_eb_pw"

    def __init__(
        self,
        qoi: PointwiseQuantityOfInterestExpression,
        type: str | ErrorBound,
        eb: int | float | str | Parameter,
        qoi_dtype: str | ToFloatMode = ToFloatMode.lossless,
    ) -> None:
        with ctx.safeguard(self):
            with ctx.parameter("qoi"):
                TypeCheckError.check_instance_or_raise(qoi, str)

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

            with ctx.parameter("qoi"):
                self._qoi = qoi
                self._qoi_expr = PointwiseQuantityOfInterest(qoi)

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

        parameters = frozenset(self._qoi_expr.late_bound_constants)

        if isinstance(self._eb, Parameter):
            parameters = parameters.union([self._eb])

        return parameters

    def evaluate_qoi(
        self,
        data: np.ndarray[S, np.dtype[T]],
        late_bound: Bindings,
    ) -> np.ndarray[S, np.dtype[F]]:
        """
        Evaluate the derived quantity of interest on the `data` in the
        floating-point data type selected by the `qoi_dtype` parameter.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Data for which the quantity of interest is evaluated.
        late_bound : Bindings
            Bindings for late-bound constants in the quantity of interest.

        Returns
        -------
        qoi : np.ndarray[S, np.dtype[F]]
            Evaluated quantity of interest, in floating-point.

        Raises
        ------
        TypeError
            if the `data` could not be losslessly cast to `qoi_dtype`.
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

        with ctx.safeguard(self):
            with ctx.parameter("qoi_dtype"):
                ftype: np.dtype[F] = self._qoi_dtype.floating_point_dtype_for(
                    data.dtype
                )  # type: ignore
                data_float: np.ndarray[tuple[int], np.dtype[F]] = to_float(
                    data, ftype=ftype
                ).flatten()

            late_bound_constants: dict[
                Parameter, np.ndarray[tuple[int], np.dtype[F]]
            ] = dict()
            with ctx.parameter("qoi"):
                for c in self._qoi_expr.late_bound_constants:
                    with ctx.parameter(c):
                        late_bound_constants[c] = (
                            late_bound.resolve_ndarray_with_lossless_cast(
                                c, data.shape, ftype
                            ).flatten()
                        )

        qoi_: np.ndarray[tuple[int], np.dtype[F]] = self._qoi_expr.eval(
            data_float,
            late_bound_constants,
        )
        qoi: np.ndarray[S, np.dtype[F]] = _reshape(qoi_, data.shape)
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
        for the quantity of interest on the `data`.

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
        TypeError
            if the `data` could not be losslessly cast to `qoi_dtype`.
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
            if the late-bound `eb` could not be broadcast to the `data`'s shape.
        ValueError
            if the late-bound `eb` is non-finite, i.e. infinite or NaN, or an
            invalid error bound value for the error bound `type`.
        """

        with ctx.safeguard(self):
            with ctx.parameter("qoi_dtype"):
                ftype: np.dtype[np.floating] = self._qoi_dtype.floating_point_dtype_for(
                    data.dtype
                )
                data_float: np.ndarray[tuple[int], np.dtype[np.floating]] = to_float(
                    data, ftype=ftype
                ).flatten()
                prediction_float: np.ndarray[tuple[int], np.dtype[np.floating]] = (
                    to_float(prediction, ftype=ftype).flatten()
                )

            late_bound_constants: dict[
                Parameter, np.ndarray[tuple[int], np.dtype[np.floating]]
            ] = dict()
            with ctx.parameter("qoi"):
                for c in self._qoi_expr.late_bound_constants:
                    with ctx.parameter(c):
                        late_bound_constants[c] = (
                            late_bound.resolve_ndarray_with_lossless_cast(
                                c, data.shape, ftype
                            ).flatten()
                        )

            # optimization: only evaluate the QoI where necessary
            if where is not True:
                data_float = np.extract(where, data_float)
                prediction_float = np.extract(where, prediction_float)
                late_bound_constants = {
                    c: np.extract(where, cv) for (c, cv) in late_bound_constants.items()
                }

            qoi_data: np.ndarray[tuple[int], np.dtype[np.floating]] = (
                self._qoi_expr.eval(data_float, late_bound_constants)
            )
            qoi_prediction: np.ndarray[tuple[int], np.dtype[np.floating]] = (
                self._qoi_expr.eval(prediction_float, late_bound_constants)
            )

            with ctx.parameter("eb"):
                eb: np.ndarray[tuple[()] | tuple[int], np.dtype[np.floating]] = (
                    late_bound.resolve_ndarray_with_saturating_finite_float_cast(
                        self._eb,
                        data.shape,
                        ftype,
                    ).flatten()
                    if isinstance(self._eb, Parameter)
                    else saturating_finite_float_cast(self._eb, ftype)
                )
                if isinstance(self._eb, Parameter):
                    _eb: Parameter = self._eb
                    with ctx.late_bound_parameter(_eb):
                        _check_error_bound(self._type, eb)

        if where is not True and eb.shape != ():
            eb = np.extract(where, eb)

        finite_ok: np.ndarray[tuple[int], np.dtype[np.bool]] = np.less_equal(
            _compute_finite_absolute_error(self._type, qoi_data, qoi_prediction),
            _compute_finite_absolute_error_bound(self._type, eb, qoi_data),
        )

        ok_: np.ndarray[tuple[int], np.dtype[np.bool]] = _ensure_array(finite_ok)
        np.equal(qoi_data, qoi_prediction, out=ok_, where=np.isinf(qoi_data))
        np.isnan(qoi_prediction, out=ok_, where=np.isnan(qoi_data))

        # the check succeeds where `where` is False
        ok: np.ndarray[S, np.dtype[np.bool]]
        if where is True:
            ok = _reshape(ok_, data.shape)
        else:
            ok = _ones(data.shape, np.dtype(np.bool))
            _place(ok, where, ok_)

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
        Compute the intervals in which the error bound is upheld with
        respect to the quantity of interest on the `data`.

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
        TypeError
            if the `data` could not be losslessly cast to `qoi_dtype`.
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
            if the late-bound `eb` could not be broadcast to the `data`'s shape.
        ValueError
            if the late-bound `eb` is non-finite, i.e. infinite or NaN, or an
            invalid error bound value for the error bound `type`.
        """

        with ctx.safeguard(self):
            with ctx.parameter("qoi_dtype"):
                ftype: np.dtype[np.floating] = self._qoi_dtype.floating_point_dtype_for(
                    data.dtype
                )
                data_float: np.ndarray[tuple[int], np.dtype[np.floating]] = to_float(
                    data, ftype=ftype
                ).flatten()

            late_bound_constants: dict[
                Parameter, np.ndarray[tuple[int], np.dtype[np.floating]]
            ] = dict()
            with ctx.parameter("qoi"):
                for c in self._qoi_expr.late_bound_constants:
                    with ctx.parameter(c):
                        late_bound_constants[c] = (
                            late_bound.resolve_ndarray_with_lossless_cast(
                                c, data.shape, ftype
                            ).flatten()
                        )

            # optimization: only evaluate the QoI and compute data bounds where
            #  necessary
            if where is not True:
                data_float = np.extract(where, data_float)
                late_bound_constants = {
                    c: np.extract(where, cv) for (c, cv) in late_bound_constants.items()
                }

            data_qoi: np.ndarray[tuple[int], np.dtype[np.floating]] = (
                self._qoi_expr.eval(data_float, late_bound_constants)
            )

            with ctx.parameter("eb"):
                eb: np.ndarray[tuple[()] | tuple[int], np.dtype[np.floating]] = (
                    late_bound.resolve_ndarray_with_saturating_finite_float_cast(
                        self._eb,
                        data.shape,
                        ftype,
                    ).flatten()
                    if isinstance(self._eb, Parameter)
                    else saturating_finite_float_cast(self._eb, ftype)
                )
                if isinstance(self._eb, Parameter):
                    _eb: Parameter = self._eb
                    with ctx.late_bound_parameter(_eb):
                        _check_error_bound(self._type, eb)

            if where is not True and eb.shape != ():
                eb = np.extract(where, eb)

        qoi_lower, qoi_upper = _apply_finite_qoi_error_bound(self._type, eb, data_qoi)

        # compute the bounds in data space
        data_float_lower_, data_float_upper_ = self._qoi_expr.compute_data_bounds(
            qoi_lower,
            qoi_upper,
            data_float,
            late_bound_constants,
        )

        # the data bounds can be arbitrary where `where` is False since they
        #  are later overridden by preserve_only_where
        data_float_lower: np.ndarray[S, np.dtype[np.floating]]
        data_float_upper: np.ndarray[S, np.dtype[np.floating]]
        if where is True:
            data_float_lower = _reshape(data_float_lower_, data.shape)
            data_float_upper = _reshape(data_float_upper_, data.shape)
        else:
            data_float_lower = np.full(data.shape, -np.inf, ftype)
            data_float_upper = np.full(data.shape, np.inf, ftype)
            _place(data_float_lower, where, data_float_lower_)
            _place(data_float_upper, where, data_float_upper_)

        wheref: Literal[True] | np.ndarray[tuple[int], np.dtype[np.bool]] = (
            True if where is True else where.flatten()
        )

        return compute_safe_data_lower_upper_interval_union(
            data, data_float_lower, data_float_upper
        ).preserve_only_where(wheref)

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

        The footprint is equivalent to `foot & where`.

        Parameters
        ----------
        foot : np.ndarray[S, np.dtype[np.bool]]
            Array for which the footprint is computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the footprint at pointwise checks where the condition
            is [`True`][True].

        Returns
        -------
        print : np.ndarray[S, np.dtype[np.bool]]
            The footprint of the `foot` array.
        """

        return _logical_and(foot, where)

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

        The inverse footprint is equivalent to `foot & where`.

        Parameters
        ----------
        foot : np.ndarray[S, np.dtype[np.bool]]
            Array for which the inverse footprint is computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the inverse footprint at pointwise checks where the
            condition is [`True`][True].

        Returns
        -------
        print : np.ndarray[S, np.dtype[np.bool]]
            The inverse footprint of the `foot` array.
        """

        return _logical_and(foot, where)

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
            type=self._type.name,
            eb=self._eb,
            qoi_dtype=self._qoi_dtype.name,
        )
