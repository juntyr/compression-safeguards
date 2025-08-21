"""
Sign-preserving safeguard.
"""

__all__ = ["SignPreservingSafeguard"]

from collections.abc import Set

import numpy as np

from ...utils._compat import _floating_smallest_subnormal
from ...utils.bindings import Bindings, Parameter
from ...utils.cast import from_total_order, lossless_cast, to_total_order
from ...utils.intervals import Interval, IntervalUnion, Lower, Upper
from ...utils.typing import S, T
from .abc import PointwiseSafeguard


class SignPreservingSafeguard(PointwiseSafeguard):
    r"""
    The `SignPreservingSafeguard` guarantees that values have the same sign
    (-1, 0, +1) in the decompressed output as they have in the input data.

    NaN values are preserved as NaN values with the same sign bit.

    This safeguard can be configured to preserve the sign relative to a custom
    `offset`, e.g. to preserve global minima and maxima. The sign is then
    defined based on arithmetic comparison, i.e.

    \[
        \text{sign}(x, offset) = \begin{cases}
            -\text{NaN} \quad &\text{if } x \text{ is} -\text{NaN} \\
            -1 \quad &\text{if } x < offset \\
            0 \quad &\text{if } x = offset \\
            +1 \quad &\text{if } x > offset \\
            +\text{NaN} \quad &\text{if } x \text{ is} +\text{NaN}
        \end{cases}
    \]

    This safeguard should be combined with e.g. an error bound, as it by itself
    accepts *any* value with the same sign.

    Parameters
    ----------
    offset : int | float | str | Parameter, optional
        The non-NaN value of or the late-bound parameter name for the offset
        compared to which the sign is computed. By default, the offset is
        zero. Values that are above / below / equal to the offset are
        guaranteed to stay above / below / equal to the offset, respectively.
        Literal values are (unsafely) cast to the data dtype before comparison.
    """

    __slots__ = ("_offset",)
    _offset: int | float | Parameter

    kind = "sign"

    def __init__(self, *, offset: int | float | str | Parameter = 0) -> None:
        if isinstance(offset, Parameter):
            self._offset = offset
        elif isinstance(offset, str):
            self._offset = Parameter(offset)
        else:
            assert isinstance(offset, int) or (not np.isnan(offset)), (
                "offset must not be NaN"
            )
            self._offset = offset

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

        return (
            frozenset([self._offset])
            if isinstance(self._offset, Parameter)
            else frozenset()
        )

    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        decoded: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check for which elements in the `decoded` array the signs match the
        signs of the `data` array elements'.

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

        offset: np.ndarray[tuple[()] | S, np.dtype[T]] = (
            late_bound.resolve_ndarray_with_lossless_cast(
                self._offset,
                data.shape,
                data.dtype,
            )
            if isinstance(self._offset, Parameter)
            else lossless_cast(self._offset, data.dtype, "sign safeguard offset")
        )
        assert np.all(~np.isnan(offset)), "offset must not contain NaNs"

        ok = np.where(
            # NaN values keep their sign bit
            np.isnan(data),
            np.isnan(decoded) & (np.signbit(data) == np.signbit(decoded)),
            np.where(
                # values equal to the offset (sign=0) stay equal
                data == offset,
                decoded == offset,
                # values below (sign=-1) stay below,
                # values above (sign=+1) stay above
                np.where(data < offset, decoded < offset, decoded > offset),
            ),
        )

        return ok  # type: ignore

    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the `data`'s sign is preserved.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Data for which the safe intervals should be computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the `data`'s sign is preserved.
        """

        offsetf: np.ndarray[tuple[()] | tuple[int], np.dtype[T]] = (
            late_bound.resolve_ndarray_with_lossless_cast(
                self._offset,
                data.shape,
                data.dtype,
            ).flatten()
            if isinstance(self._offset, Parameter)
            else lossless_cast(self._offset, data.dtype, "sign safeguard offset")
        )
        assert np.all(~np.isnan(offsetf)), "offset must not contain NaNs"
        offsetf_total: np.ndarray[
            tuple[()] | tuple[int], np.dtype[np.unsignedinteger]
        ] = to_total_order(offsetf)

        with np.errstate(over="ignore", under="ignore"):
            if np.issubdtype(data.dtype, np.floating):
                # special case for floating point -0.0 / +0.0 which both have
                #  zero sign and thus have weird below / above intervals
                smallest_subnormal = _floating_smallest_subnormal(data.dtype)  # type: ignore
                below_upper = np.array(
                    np.where(
                        offsetf == 0,
                        -smallest_subnormal,
                        from_total_order(offsetf_total - 1, data.dtype),
                    )
                )
                above_lower = np.array(
                    np.where(
                        offsetf == 0,
                        smallest_subnormal,
                        from_total_order(offsetf_total + 1, data.dtype),
                    )
                )
            else:
                below_upper = np.array(from_total_order(offsetf_total - 1, data.dtype))
                above_lower = np.array(from_total_order(offsetf_total + 1, data.dtype))

        dataf = data.flatten()
        valid = (
            Interval.empty_like(dataf)
            .preserve_signed_nan(dataf, equal_nan=True)
            .preserve_non_nan(dataf)
        )

        # preserve zero-sign values exactly
        Lower(dataf) <= valid[dataf == offsetf] <= Upper(dataf)
        valid[dataf < offsetf] <= Upper(below_upper)
        Lower(above_lower) <= valid[dataf > offsetf]

        return valid.into_union()

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(kind=type(self).kind, offset=self._offset)
