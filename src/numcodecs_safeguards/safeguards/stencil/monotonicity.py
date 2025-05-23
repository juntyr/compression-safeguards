"""
Monotonicity-preserving safeguard.
"""

__all__ = ["Monotonicity", "MonotonicityPreservingSafeguard"]

from enum import Enum
from operator import ge, gt, le, lt

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ...cast import _isfinite, _isnan, from_total_order, to_total_order
from ...intervals import Interval, IntervalUnion, Lower, Upper
from . import BoundaryCondition, NeighbourhoodAxis, _pad_with_boundary
from .abc import S, StencilSafeguard, T

_STRICT = ((lt, gt, False, False),) * 2
_STRICT_WITH_CONSTS = ((lt, gt, True, False),) * 2
_STRICT_TO_WEAK = ((lt, gt, False, False), (le, ge, True, True))
_WEAK = ((le, ge, False, True), (le, ge, True, True))


class Monotonicity(Enum):
    """
    Different levels of monotonicity that can be enforced by the
    [`MonotonicityPreservingSafeguard`][numcodecs_safeguards.safeguards.stencil.monotonicity.MonotonicityPreservingSafeguard].
    """

    strict = _STRICT
    """
    Strictly increasing/decreasing sequences in the input array are guaranteed
    to be strictly increasing/decreasing in the decoded array.

    Sequences that are not strictly increasing/decreasing or contain non-finite
    values are not affected.
    """

    strict_with_consts = _STRICT_WITH_CONSTS
    """
    Strictly increasing/decreasing/constant sequences in the input array are
    guaranteed to be strictly increasing/decreasing/constant in the decoded
    array.

    Sequences that are not strictly increasing/decreasing/constant or contain
    non-finite values are not affected.
    """

    strict_to_weak = _STRICT_TO_WEAK
    """
    Strictly increasing/decreasing sequences in the input array are guaranteed
    to be *weakly* increasing/decreasing (or constant) in the decoded array.

    Sequences that are not strictly increasing/decreasing or contain non-finite
    values are not affected.
    """

    weak = _WEAK
    """
    Weakly increasing/decreasing (but not constant) sequences in the input
    array are guaranteed to be weakly increasing/decreasing (or constant) in
    the decoded array.

    Sequences that are not weakly increasing/decreasing or are constant or
    contain non-finite values are not affected.
    """


class MonotonicityPreservingSafeguard(StencilSafeguard):
    r"""
    The `MonotonicityPreservingSafeguard` guarantees that sequences that are
    monotonic in the input are guaranteed to be monotonic in the decompressed
    output.

    Monotonic sequences are detected using per-axis moving windows with a
    constant symmetric size of $(1 + window \cdot 2)$. Typically, the window
    size should be chosen to be large enough to ignore noise, i.e. $>1$, but
    small enough to capture details.

    The safeguard supports enforcing four levels of
    [`Monotonicity`][numcodecs_safeguards.safeguards.stencil.monotonicity.Monotonicity]:
    `strict`, `strict_with_consts`, `strict_to_weak`, `weak`.

    Windows that are not monotonic or contain non-finite data are skipped. Axes
    that have fewer elements than the window size are skipped as well.

    Parameters
    ----------
    monotonicity : str | Monotonicity
        The level of monotonicity that is guaranteed to be preserved by the
        safeguard.
    window : int
        Positive symmetric half-window size; the window has size
        $(1 + window \cdot 2)$.
    boundary : str | BoundaryCondition
        Boundary condition for evaluating the monotonicity near the data array
        domain boundaries, e.g. by extending values.
    constant_boundary : None | int | float
        Optional constant value with which the data array domain is extended
        for a constant boundary. The value must be safely convertable (without
        over- or underflow or invalid values) to the data type.
    axis : None | int
        The axis along which the monotonicity is preserved. The default,
        [`None`][None], is to preserve along all axes.
    """

    __slots__ = ("_monotonicity", "_window", "_boundary", "_constant_boundary", "_axis")
    _monotonicity: Monotonicity
    _window: int
    _boundary: BoundaryCondition
    _constant_boundary: None | int | float
    _axis: None | int

    kind = "monotonicity"

    def __init__(
        self,
        monotonicity: str | Monotonicity,
        window: int,
        boundary: str | BoundaryCondition,
        constant_boundary: None | int | float = None,
        axis: None | int = None,
    ):
        self._monotonicity = (
            monotonicity
            if isinstance(monotonicity, Monotonicity)
            else Monotonicity[monotonicity]
        )

        assert window > 0, "window size must be positive"
        self._window = window

        self._boundary = (
            boundary
            if isinstance(boundary, BoundaryCondition)
            else BoundaryCondition[boundary]
        )
        assert (self._boundary != BoundaryCondition.constant) == (
            constant_boundary is None
        ), (
            "constant_boundary must be provided if and only if the constant boundary condition is used"
        )
        self._constant_boundary = constant_boundary

        self._axis = axis

    def compute_check_neighbourhood_for_data_shape(
        self, data_shape: tuple[int, ...]
    ) -> tuple[None | NeighbourhoodAxis, ...]:
        """
        Compute the shape of the data neighbourhood for data of a given shape.
        [`None`][None] is returned along dimensions for which there is no data
        neighbourhood.

        This method also checks that the data shape is compatible with this
        stencil safeguard.

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

        for axis, alen in enumerate(data_shape):
            if (
                self._axis is not None
                and axis != self._axis
                and axis != (len(data_shape) + self._axis)
            ):
                continue

            if alen == 0:
                continue

            neighbourhood[axis] = NeighbourhoodAxis(
                self._window,
                self._window,
            )

        return tuple(neighbourhood)

    def check_pointwise(
        self, data: np.ndarray[S, T], decoded: np.ndarray[S, T]
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which monotonic sequences centred on the points in the `data`
        array are preserved in the `decoded` array.

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.

        Returns
        -------
        ok : np.ndarray
            Pointwise, `True` if the check succeeded for this element.
        """

        ok = np.ones_like(data, dtype=np.bool)

        window = 1 + self._window * 2

        for axis, alen in enumerate(data.shape):
            if (
                self._axis is not None
                and axis != self._axis
                and axis != (data.ndim + self._axis)
            ):
                continue

            if self._boundary == BoundaryCondition.valid and alen < window:
                continue

            if alen == 0:
                continue

            data_boundary = _pad_with_boundary(
                data,
                self._boundary,
                self._window,
                self._window,
                self._constant_boundary,
                axis,
            )
            decoded_boundary = _pad_with_boundary(
                decoded,
                self._boundary,
                self._window,
                self._window,
                self._constant_boundary,
                axis,
            )

            data_windows = sliding_window_view(data_boundary, window, axis=axis)
            decoded_windows = sliding_window_view(decoded_boundary, window, axis=axis)

            data_monotonic = self._monotonic_sign(data_windows, is_decoded=False)
            decoded_monotonic = self._monotonic_sign(decoded_windows, is_decoded=True)

            # for monotonic windows, check that the monotonicity matches
            axis_ok = self._monotonic_sign_not_equal(data_monotonic, decoded_monotonic)

            if self._boundary == BoundaryCondition.valid:
                s = tuple(
                    [slice(None)] * axis
                    + [slice(self._window, -self._window)]
                    + [slice(None)] * (data.ndim - axis - 1)
                )
            else:
                s = tuple([slice(None)] * data.ndim)

            ok[s] &= ~axis_ok.reshape(ok[s].shape)

        return ok  # type: ignore

    def compute_safe_intervals(
        self, data: np.ndarray[S, T]
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the monotonicity of the `data` is
        preserved.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the monotonicity is preserved.
        """

        window = 1 + self._window * 2

        (_lt, _gt, _eq, is_weak) = self._monotonicity.value[1]
        nudge = 0 if is_weak else 1

        valid = Interval.full_like(data)

        # track which elements have any monotonicity-based restrictions
        #  imposed upon them
        any_restriction = np.zeros_like(data, dtype=np.bool)

        for axis, alen in enumerate(data.shape):
            if (
                self._axis is not None
                and axis != self._axis
                and axis != (data.ndim + self._axis)
            ):
                continue

            if self._boundary == BoundaryCondition.valid and alen < window:
                continue

            if alen == 0:
                continue

            data_boundary = _pad_with_boundary(
                data,
                self._boundary,
                self._window,
                self._window,
                self._constant_boundary,
                axis,
            )
            data_windows = sliding_window_view(data_boundary, window, axis=axis)
            data_monotonic = self._monotonic_sign(data_windows, is_decoded=False)

            # compute, pointwise, if the element has a decreasing (lt),
            #  increasing (gt), or equality (eq) constraint imposed upon it
            # for the lt and gt variants, compute both a mask for accesses to
            #  elements on the left and to elements on the right, since the
            #  elements at the left and right edge of the window cannot access
            #  one further element to the left / right, respectively
            elem_lt_left, elem_lt_right, elem_eq, elem_gt_left, elem_gt_right = (
                np.zeros_like(data_boundary, dtype=np.bool),
                np.zeros_like(data_boundary, dtype=np.bool),
                np.zeros_like(data_boundary, dtype=np.bool),
                np.zeros_like(data_boundary, dtype=np.bool),
                np.zeros_like(data_boundary, dtype=np.bool),
            )
            for w in range(window):
                # ensure that all members of the window receive their
                #  monotonicity contribution
                s = tuple(
                    [slice(None)] * axis
                    + [slice(w, None if (w + 1) == window else -window + w + 1)]
                    + [slice(None)] * (data_boundary.ndim - axis - 1)
                )

                if w > 0:
                    elem_lt_left[s] |= data_monotonic[..., 0] == -1
                if w < (window - 1):
                    elem_lt_right[s] |= data_monotonic[..., 0] == -1

                elem_eq[s] |= data_monotonic[..., 0] == 0

                if w > 0:
                    elem_gt_left[s] |= data_monotonic[..., 0] == +1
                if w < (window - 1):
                    elem_gt_right[s] |= data_monotonic[..., 0] == +1

            if self._boundary == BoundaryCondition.valid:
                s = tuple([slice(None)] * data.ndim)
            else:
                s = tuple(
                    [slice(None)] * axis
                    + [slice(self._window, -self._window)]
                    + [slice(None)] * (data_boundary.ndim - axis - 1)
                )

            # if any element has an equality constraint, impose it and
            #  intersect with the overall valid interval
            if np.any(elem_eq):
                any_restriction |= elem_eq[s]
                valid_eq = Interval.full_like(data)
                Lower(data.flatten()) <= valid_eq[elem_eq[s].flatten()] <= Upper(
                    data.flatten()
                )
                valid = valid.intersect(valid_eq)

            # if any element has a decreasing constraint, impose it and
            #  intersect with the overall valid interval
            # constraints that need an element to the left/right are imposed
            #  separately
            # nudge the lower bound up and upper bound down for strict
            #  monotonicity to ensure that the safe intervals don't overlap
            if np.any(elem_lt_left | elem_lt_right):
                valid_lt = Interval.full_like(data_boundary)
            if np.any(elem_lt_right):
                Lower(
                    from_total_order(
                        to_total_order(np.roll(data_boundary, -1, axis=axis)) + nudge,
                        dtype=data.dtype,
                    ).flatten()
                ) <= valid_lt[elem_lt_right.flatten()]
            if np.any(elem_lt_left):
                valid_lt[elem_lt_left.flatten()] <= Upper(
                    from_total_order(
                        to_total_order(np.roll(data_boundary, +1, axis=axis)) - nudge,
                        dtype=data.dtype,
                    ).flatten()
                )
            if np.any(elem_lt_left | elem_lt_right):
                # temporarily shape the interval bounds like the data
                #  for easier indexing
                valid_lt._lower = valid_lt._lower.reshape(data_boundary.shape)
                valid_lt._upper = valid_lt._upper.reshape(data_boundary.shape)

                if self._boundary in (
                    BoundaryCondition.reflect,
                    BoundaryCondition.symmetric,
                    BoundaryCondition.wrap,
                ):
                    # requirements inside the boundary need to be connected
                    #  back to the original data elements
                    boundary_indices = _pad_with_boundary(
                        np.arange(alen),
                        self._boundary,
                        self._window,
                        self._window,
                        None,
                        0,
                    )
                    for w in [ws for w in range(self._window) for ws in [w, -w - 1]]:
                        s_boundary = tuple(
                            [slice(None)] * axis
                            + [w]
                            + [slice(None)] * (data_boundary.ndim - axis - 1)
                        )
                        s_inner = tuple(
                            [slice(None)] * axis
                            + [boundary_indices[w] + self._window]
                            + [slice(None)] * (data_boundary.ndim - axis - 1)
                        )
                        # map the boundary values for the requirement masks and
                        #  interval back to the inner values
                        elem_lt_left[s_inner] |= elem_lt_left[s_boundary]
                        elem_lt_right[s_inner] |= elem_lt_right[s_boundary]
                        valid_lt._lower[s_inner] = from_total_order(
                            np.maximum(
                                to_total_order(valid_lt._lower[s_inner]),
                                to_total_order(valid_lt._lower[s_boundary]),
                            ),
                            dtype=data.dtype,
                        )
                        valid_lt._upper[s_inner] = from_total_order(
                            np.minimum(
                                to_total_order(valid_lt._upper[s_inner]),
                                to_total_order(valid_lt._upper[s_boundary]),
                            ),
                            dtype=data.dtype,
                        )

                any_restriction |= elem_lt_left[s] | elem_lt_right[s]
                valid = valid.intersect(
                    Interval(
                        _lower=valid_lt._lower[s].flatten(),
                        _upper=valid_lt._upper[s].flatten(),
                    )
                )

            # if any element has an increasing constraint, impose it and
            #  intersect with the overall valid interval
            # constraints that need an element to the left/right are imposed
            #  separately
            # nudge the lower bound up and upper bound down for strict
            #  monotonicity to ensure that the safe intervals don't overlap
            if np.any(elem_gt_left | elem_gt_right):
                valid_gt = Interval.full_like(data_boundary)
            if np.any(elem_gt_left):
                Lower(
                    from_total_order(
                        to_total_order(np.roll(data_boundary, +1, axis=axis)) + nudge,
                        dtype=data.dtype,
                    ).flatten()
                ) <= valid_gt[elem_gt_left.flatten()]
            if np.any(elem_gt_right):
                valid_gt[elem_gt_right.flatten()] <= Upper(
                    from_total_order(
                        to_total_order(np.roll(data_boundary, -1, axis=axis)) - nudge,
                        dtype=data.dtype,
                    ).flatten()
                )
            if np.any(elem_gt_left | elem_gt_right):
                # temporarily shape the interval bounds like the data
                #  for easier indexing
                valid_gt._lower = valid_gt._lower.reshape(data_boundary.shape)
                valid_gt._upper = valid_gt._upper.reshape(data_boundary.shape)

                if self._boundary in (
                    BoundaryCondition.reflect,
                    BoundaryCondition.symmetric,
                    BoundaryCondition.wrap,
                ):
                    # requirements inside the boundary need to be connected
                    #  back to the original data elements
                    boundary_indices = _pad_with_boundary(
                        np.arange(alen),
                        self._boundary,
                        self._window,
                        self._window,
                        None,
                        0,
                    )
                    for w in [ws for w in range(self._window) for ws in [w, -w - 1]]:
                        s_boundary = tuple(
                            [slice(None)] * axis
                            + [w]
                            + [slice(None)] * (data_boundary.ndim - axis - 1)
                        )
                        s_inner = tuple(
                            [slice(None)] * axis
                            + [boundary_indices[w] + self._window]
                            + [slice(None)] * (data_boundary.ndim - axis - 1)
                        )
                        # map the boundary values for the requirement masks and
                        #  interval back to the inner values
                        elem_gt_left[s_inner] |= elem_gt_left[s_boundary]
                        elem_gt_right[s_inner] |= elem_gt_right[s_boundary]
                        valid_gt._lower[s_inner] = from_total_order(
                            np.maximum(
                                to_total_order(valid_gt._lower[s_inner]),
                                to_total_order(valid_gt._lower[s_boundary]),
                            ),
                            dtype=data.dtype,
                        )
                        valid_gt._upper[s_inner] = from_total_order(
                            np.minimum(
                                to_total_order(valid_gt._upper[s_inner]),
                                to_total_order(valid_gt._upper[s_boundary]),
                            ),
                            dtype=data.dtype,
                        )

                any_restriction |= elem_gt_left[s] | elem_gt_right[s]
                valid = valid.intersect(
                    Interval(
                        _lower=valid_gt._lower[s].flatten(),
                        _upper=valid_gt._upper[s].flatten(),
                    )
                )

        # produce conservative safe intervals by computing the midpoint between
        #  the data and the lower/upper bound
        # for strict monotonicity, the lower bound is nudged up to ensure its
        #  midpoint rounds up while the limiting element's corresponding upper
        #  bound will round down
        lt: np.ndarray = to_total_order(valid._lower)
        ut: np.ndarray = to_total_order(valid._upper)
        dt: np.ndarray = to_total_order(data.flatten())
        if not is_weak:
            lt = np.where((lt + 1) > 0, lt + 1, lt)

        # Hacker's Delight's algorithm to compute (a + b) / 2:
        #  ((a ^ b) >> 1) + (a & b)
        valid._lower = from_total_order(((lt ^ dt) >> 1) + (lt & dt), data.dtype)
        valid._upper = from_total_order(((ut ^ dt) >> 1) + (ut & dt), data.dtype)

        # ensure that finite values remain finite since they can otherwise
        #  invalidate the monotonicity of their window
        valid = valid.intersect(
            Interval.full_like(data).preserve_finite(data.flatten())
        )

        # only apply the safe interval restrictions to elements onto which any
        #  constraints were actually imposed, i.e. leave elements alone that
        #  are not part of any monotonic window
        filtered_valid = Interval.full_like(data)
        Lower(valid._lower) <= filtered_valid[any_restriction.flatten()] <= Upper(
            valid._upper
        )

        # special case for +0.0 and -0.0:
        # - they compare equal but are ordered in binary
        # - monotonicity interval can end up empty as [+0.0, -0.0]
        # so explicitly add restricted zero values to the valid interval
        zero_valid = Interval.empty_like(data)
        Lower(data.flatten()) <= zero_valid[
            (data.flatten() == 0) & any_restriction.flatten()
        ] <= Upper(data.flatten())

        return filtered_valid.union(zero_valid)

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        config = dict(
            kind=type(self).kind,
            monotonicity=self._monotonicity.name,
            window=self._window,
            boundary=self._boundary.name,
            constant_boundary=self._constant_boundary,
            axis=self._axis,
        )

        if self._constant_boundary is None:
            del config["constant_boundary"]

        return config

    def _monotonic_sign(
        self,
        x: np.ndarray,
        *,
        is_decoded: bool,
    ) -> np.ndarray:
        (lt, gt, eq, _is_weak) = self._monotonicity.value[int(is_decoded)]

        # default to NaN
        monotonic = np.empty(x.shape[:-1])
        monotonic.fill(np.nan)

        # use comparison instead of diff to account for uints

        # +1: all(x[i+1] > x[i])
        monotonic = np.where(
            np.all(gt(x[..., 1:], x[..., :-1]), axis=-1), +1, monotonic
        )
        # -1: all(x[i+1] < x[i])
        monotonic = np.where(
            np.all(lt(x[..., 1:], x[..., :-1]), axis=-1), -1, monotonic
        )

        # 0/NaN: all(x[i+1] == x[i])
        monotonic = np.where(
            np.all(x[..., 1:] == x[..., :-1], axis=-1), 0 if eq else np.nan, monotonic
        )

        # non-finite values cannot participate in monotonic sequences
        # NaN: any(!isfinite(x[i]))
        monotonic = np.where(np.all(_isfinite(x), axis=-1), monotonic, np.nan)

        # return the result in a shape that's broadcastable to x
        return monotonic[..., np.newaxis]

    def _monotonic_sign_not_equal(
        self, data_monotonic: np.ndarray, decoded_monotonic: np.ndarray
    ) -> np.ndarray:
        match self._monotonicity:
            case Monotonicity.strict | Monotonicity.strict_with_consts:
                return np.where(
                    _isfinite(data_monotonic),
                    decoded_monotonic != data_monotonic,
                    False,
                )
            case Monotonicity.strict_to_weak | Monotonicity.weak:
                return np.where(
                    _isfinite(data_monotonic),
                    # having the opposite sign or no sign are both not equal
                    (decoded_monotonic == -data_monotonic) | _isnan(decoded_monotonic),
                    False,
                )
