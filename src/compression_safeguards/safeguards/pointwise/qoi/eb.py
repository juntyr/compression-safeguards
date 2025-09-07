"""
Pointwise quantity of interest (QoI) error bound safeguard.
"""

__all__ = ["PointwiseQuantityOfInterestErrorBoundSafeguard"]

from collections.abc import Set

import numpy as np

from ....utils.bindings import Bindings, Parameter
from ....utils.cast import saturating_finite_float_cast, to_float
from ....utils.intervals import IntervalUnion
from ....utils.typing import F, S, T
from ..._qois import PointwiseQuantityOfInterest
from ..._qois.interval import compute_safe_data_lower_upper_interval_union
from ...eb import (
    ErrorBound,
    _apply_finite_qoi_error_bound,
    _check_error_bound,
    _compute_finite_absolute_error,
    _compute_finite_absolute_error_bound,
)
from ...qois import (
    PointwiseQuantityOfInterestExpression,
    QuantityOfInterestEvaluationDType,
)
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
    decoded value produces the exact same infinite value. For a NaN quantity of
    interest, this safeguard guarantees that the quantity of interest on the
    decoded value is also NaN, but does not guarantee that it has the same
    bit pattern.

    The error bound can be verified by evaluating the QoI in the floating-point
    data type selected by `dtype` parameter using the
    [`evaluate_qoi`][compression_safeguards.safeguards.pointwise.qoi.eb.PointwiseQuantityOfInterestErrorBoundSafeguard.evaluate_qoi]
    method.

    Please refer to the
    [`PointwiseQuantityOfInterestExpression`][compression_safeguards.safeguards.qois.PointwiseQuantityOfInterestExpression]
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
    dtype : str | QuantityOfInterestEvaluationDType
        The floating-point data type in which the quantity of interest is
        evaluated. By default, the smallest floating-point data type that can
        losslessly represent all input data values is chosen.
    """

    __slots__ = (
        "_qoi",
        "_type",
        "_eb",
        "_dtype",
        "_qoi_expr",
    )
    _qoi: PointwiseQuantityOfInterestExpression
    _type: ErrorBound
    _eb: int | float | Parameter
    _dtype: QuantityOfInterestEvaluationDType
    _qoi_expr: PointwiseQuantityOfInterest

    kind = "qoi_eb_pw"

    def __init__(
        self,
        qoi: PointwiseQuantityOfInterestExpression,
        type: str | ErrorBound,
        eb: int | float | str | Parameter,
        dtype: str
        | QuantityOfInterestEvaluationDType = QuantityOfInterestEvaluationDType.lossless,
    ) -> None:
        self._type = type if isinstance(type, ErrorBound) else ErrorBound[type]

        if isinstance(eb, Parameter):
            self._eb = eb
        elif isinstance(eb, str):
            self._eb = Parameter(eb)
        else:
            _check_error_bound(self._type, eb)
            self._eb = eb

        self._dtype = (
            dtype
            if isinstance(dtype, QuantityOfInterestEvaluationDType)
            else QuantityOfInterestEvaluationDType[dtype]
        )

        try:
            qoi_expr = PointwiseQuantityOfInterest(qoi)
        except Exception as err:
            raise AssertionError(
                f"failed to parse pointwise QoI expression {qoi!r}: {err}"
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
        floating-point data type selected by the `dtype` parameter.

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
        """

        ftype: np.dtype[F] = self._dtype.floating_point_dtype_for(data.dtype)  # type: ignore
        data_float: np.ndarray[S, np.dtype[F]] = to_float(data, ftype=ftype)

        late_bound_constants: dict[Parameter, np.ndarray[S, np.dtype[F]]] = {
            c: late_bound.resolve_ndarray_with_lossless_cast(
                c, data.shape, data_float.dtype
            )
            for c in self._qoi_expr.late_bound_constants
        }

        return self._qoi_expr.eval(
            data_float,
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
        the quantity of interest on the `data`.

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

        ftype: np.dtype[np.floating] = self._dtype.floating_point_dtype_for(data.dtype)
        data_float: np.ndarray[S, np.dtype[np.floating]] = to_float(data, ftype=ftype)

        late_bound_constants: dict[Parameter, np.ndarray[S, np.dtype[np.floating]]] = {
            c: late_bound.resolve_ndarray_with_lossless_cast(
                c, data.shape, data_float.dtype
            )
            for c in self._qoi_expr.late_bound_constants
        }

        qoi_data = self._qoi_expr.eval(data_float, late_bound_constants)
        qoi_decoded = self._qoi_expr.eval(
            to_float(decoded, ftype=ftype), late_bound_constants
        )

        eb: np.ndarray[tuple[()] | S, np.dtype[np.floating]] = (
            late_bound.resolve_ndarray_with_saturating_finite_float_cast(
                self._eb,
                qoi_data.shape,
                qoi_data.dtype,
            )
            if isinstance(self._eb, Parameter)
            else saturating_finite_float_cast(
                self._eb, qoi_data.dtype, "pointwise QoI error bound safeguard eb"
            )
        )
        _check_error_bound(self._type, eb)

        finite_ok: np.ndarray[S, np.dtype[np.bool]] = np.less_equal(
            _compute_finite_absolute_error(self._type, qoi_data, qoi_decoded),
            _compute_finite_absolute_error_bound(self._type, eb, qoi_data),
        )

        ok: np.ndarray[S, np.dtype[np.bool]] = np.array(finite_ok, copy=None)  # type: ignore
        np.copyto(ok, qoi_data == qoi_decoded, where=np.isinf(qoi_data), casting="no")
        np.copyto(ok, np.isnan(qoi_decoded), where=np.isnan(qoi_data), casting="no")

        return ok

    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
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

        Returns
        -------
        intervals : IntervalUnion[T, int, int]
            Union of intervals in which the error bound is upheld.
        """

        ftype: np.dtype[np.floating] = self._dtype.floating_point_dtype_for(data.dtype)
        data_float: np.ndarray[S, np.dtype[np.floating]] = to_float(data, ftype=ftype)

        late_bound_constants: dict[Parameter, np.ndarray[S, np.dtype[np.floating]]] = {
            c: late_bound.resolve_ndarray_with_lossless_cast(
                c, data.shape, data_float.dtype
            )
            for c in self._qoi_expr.late_bound_constants
        }

        data_qoi = self._qoi_expr.eval(data_float, late_bound_constants)

        eb: np.ndarray[tuple[()] | S, np.dtype[np.floating]] = (
            late_bound.resolve_ndarray_with_saturating_finite_float_cast(
                self._eb,
                data_qoi.shape,
                data_qoi.dtype,
            )
            if isinstance(self._eb, Parameter)
            else saturating_finite_float_cast(
                self._eb, data_qoi.dtype, "pointwise QoI error bound safeguard eb"
            )
        )
        _check_error_bound(self._type, eb)

        qoi_lower_upper: tuple[
            np.ndarray[S, np.dtype[np.floating]], np.ndarray[S, np.dtype[np.floating]]
        ] = _apply_finite_qoi_error_bound(
            self._type,
            eb,
            data_qoi,
        )
        qoi_lower, qoi_upper = qoi_lower_upper

        # compute the bounds in data space
        data_float_lower, data_float_upper = self._qoi_expr.compute_data_bounds(
            qoi_lower,
            qoi_upper,
            data_float,
            late_bound_constants,
        )

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
            type=self._type.name,
            eb=self._eb,
            dtype=self._dtype.name,
        )
