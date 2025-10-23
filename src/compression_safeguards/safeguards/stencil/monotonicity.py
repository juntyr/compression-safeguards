"""
Monotonicity-preserving safeguard.
"""

__all__ = ["Monotonicity", "MonotonicityPreservingSafeguard"]

from collections.abc import Set
from enum import Enum
from operator import ge, gt, le, lt
from typing import ClassVar

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing_extensions import (
    assert_never,  # MSPV 3.11
    override,  # MSPV 3.12
)

from ...utils._compat import (
    _ensure_array,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
    _ones,
    _zeros,
)
from ...utils.bindings import Bindings, Parameter
from ...utils.cast import from_total_order, lossless_cast, to_total_order
from ...utils.error import TypeCheckError, ctx, lookup_enum_or_raise
from ...utils.intervals import Interval, IntervalUnion, Lower, Upper
from ...utils.typing import JSON, S, T
from . import BoundaryCondition, NeighbourhoodAxis, _pad_with_boundary
from .abc import StencilSafeguard

_STRICT = ((lt, gt, False, False),) * 2
_STRICT_WITH_CONSTS = ((lt, gt, True, False),) * 2
_STRICT_TO_WEAK = ((lt, gt, False, False), (le, ge, True, True))
_WEAK = ((le, ge, False, True), (le, ge, True, True))


class Monotonicity(Enum):
    """
    Different levels of monotonicity that can be preserved by the
    [`MonotonicityPreservingSafeguard`][compression_safeguards.safeguards.stencil.monotonicity.MonotonicityPreservingSafeguard].
    """

    strict = _STRICT
    """
    Strictly increasing/decreasing sequences in the input array are guaranteed
    to be strictly increasing/decreasing in the corrected array.

    Sequences that are not strictly increasing/decreasing or contain NaN values
    are not affected.
    """

    strict_with_consts = _STRICT_WITH_CONSTS
    """
    Strictly increasing/decreasing/constant sequences in the input array are
    guaranteed to be strictly increasing/decreasing/constant in the corrected
    array.

    Sequences that are not strictly increasing/decreasing/constant or contain
    NaN values are not affected.
    """

    strict_to_weak = _STRICT_TO_WEAK
    """
    Strictly increasing/decreasing sequences in the input array are guaranteed
    to be *weakly* increasing/decreasing (or constant) in the corrected array.

    Sequences that are not strictly increasing/decreasing or contain NaN values
    are not affected.
    """

    weak = _WEAK
    """
    Weakly increasing/decreasing (but not constant) sequences in the input
    array are guaranteed to be weakly increasing/decreasing (or constant) in
    the corrected array.

    Sequences that are not weakly increasing/decreasing or are constant or
    contain NaN values are not affected.
    """


class MonotonicityPreservingSafeguard(StencilSafeguard):
    r"""
    The `MonotonicityPreservingSafeguard` guarantees that sequences that are
    monotonic in the input are guaranteed to be monotonic in the decompressed
    output.

    Monotonic sequences are detected using arithmetic comparisons in per-axis
    moving windows with a constant symmetric size of $(1 + window \cdot 2)$.
    Typically, the window size should be chosen to be large enough to ignore
    noise, i.e. $>1$, but small enough to capture details.

    The safeguard supports enforcing four levels of
    [`Monotonicity`][compression_safeguards.safeguards.stencil.monotonicity.Monotonicity]:
    `strict`, `strict_with_consts`, `strict_to_weak`, `weak`.

    Windows that are not monotonic or contain NaN values are skipped.

    If the provided `axis` index is out of range for some data shape, the
    safeguard is not applied to that data.

    If the
    [valid][compression_safeguards.safeguards.stencil.BoundaryCondition.valid]
    `boundary` condition is used, axes that have fewer elements than
    $(1 + window \cdot 2)$ are skipped. Using a different
    [`BoundaryCondition`][compression_safeguards.safeguards.stencil.BoundaryCondition]
    ensures that the safeguard is always applied.

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
    constant_boundary : None | int | float | str | Parameter
        The optional value of or the late-bound parameter name for the constant
        value with which the data array domain is extended for a constant
        boundary. The value must be losslessly convertible to the data dtype.
    axis : None | int
        The axis along which the monotonicity is preserved. The default,
        [`None`][None], is to preserve along all axes.

    Raises
    ------
    TypeCheckError
        if any parameter has the wrong type.
    ValueError
        if `window` is not positive.
    ValueError
        if `boundary` does not name a valid boundary condition variant.
    ValueError
        if `constant_boundary` is, not, provided if and only if the `boundary`
        is constant.
    ValueError
        if `constant_boundary` uses the non-scalar `$x` or `$X` late-bound
        parameters.
    """

    __slots__: tuple[str, ...] = (
        "_monotonicity",
        "_window",
        "_boundary",
        "_constant_boundary",
        "_axis",
    )
    _monotonicity: Monotonicity
    _window: int
    _boundary: BoundaryCondition
    _constant_boundary: None | int | float | Parameter
    _axis: None | int

    kind: ClassVar[str] = "monotonicity"

    def __init__(
        self,
        monotonicity: str | Monotonicity,
        window: int,
        boundary: str | BoundaryCondition,
        constant_boundary: None | int | float | str | Parameter = None,
        axis: None | int = None,
    ) -> None:
        with ctx.safeguard(self):
            with ctx.parameter("monotonicity"):
                TypeCheckError.check_instance_or_raise(monotonicity, str | Monotonicity)
                self._monotonicity = (
                    monotonicity
                    if isinstance(monotonicity, Monotonicity)
                    else lookup_enum_or_raise(Monotonicity, monotonicity)
                )

            with ctx.parameter("window"):
                TypeCheckError.check_instance_or_raise(window, int)
                if window <= 0:
                    raise ValueError("must be positive") | ctx
                self._window = window

            with ctx.parameter("boundary"):
                TypeCheckError.check_instance_or_raise(
                    boundary, str | BoundaryCondition
                )
                self._boundary = (
                    boundary
                    if isinstance(boundary, BoundaryCondition)
                    else lookup_enum_or_raise(BoundaryCondition, boundary)
                )

            with ctx.parameter("constant_boundary"):
                TypeCheckError.check_instance_or_raise(
                    constant_boundary, None | int | float | str | Parameter
                )

                if (self._boundary != BoundaryCondition.constant) != (
                    constant_boundary is None
                ):
                    raise (
                        ValueError(
                            "must be provided if and only if the constant boundary condition is used"
                        )
                        | ctx
                    )

                if isinstance(constant_boundary, Parameter):
                    self._constant_boundary = constant_boundary
                elif isinstance(constant_boundary, str):
                    self._constant_boundary = Parameter(constant_boundary)
                else:
                    self._constant_boundary = constant_boundary

                if isinstance(
                    self._constant_boundary, Parameter
                ) and self._constant_boundary in ["$x", "$X"]:
                    raise (
                        ValueError(
                            f"must be scalar but late-bound constant data {self._constant_boundary} may not be"
                        )
                        | ctx
                    )

            with ctx.parameter("axis"):
                TypeCheckError.check_instance_or_raise(axis, None | int)
                self._axis = axis

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

        return (
            frozenset([self._constant_boundary])
            if isinstance(self._constant_boundary, Parameter)
            else frozenset()
        )

    @override
    def compute_check_neighbourhood_for_data_shape(
        self, data_shape: tuple[int, ...]
    ) -> tuple[dict[BoundaryCondition, NeighbourhoodAxis], ...]:
        """
        Compute the shape of the data neighbourhood for data of a given shape.

        An empty [`dict`][dict] is returned along dimensions where the
        monotonicity safeguard is not applied.

        Parameters
        ----------
        data_shape : tuple[int, ...]
            The shape of the data.

        Returns
        -------
        neighbourhood_shape : tuple[dict[BoundaryCondition, NeighbourhoodAxis], ...]
            The shape of the data neighbourhood.
        """

        neighbourhood: list[dict[BoundaryCondition, NeighbourhoodAxis]] = [
            dict() for _ in data_shape
        ]

        for axis, alen in enumerate(data_shape):
            if (
                self._axis is not None
                and axis != self._axis
                and axis != (len(data_shape) + self._axis)
            ):
                continue

            if alen == 0:
                continue

            neighbourhood[axis][self._boundary] = NeighbourhoodAxis(
                self._window,
                self._window,
            )

        return tuple(neighbourhood)

    @override
    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which monotonic sequences centred on the points in the `data`
        array are preserved in the `prediction` array.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Original data array, relative to which the `prediction` is checked.
        prediction : np.ndarray[S, np.dtype[T]]
            Prediction for the `data` array.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        ok : np.ndarray[S, np.dtype[np.bool]]
            Pointwise, `True` if the check succeeded for this element.

        Raises
        ------
        LateBoundParameterResolutionError
            if the `constant_boundary` is late-bound but its late-bound
            parameter is not in `late_bound`.
        ValueError
            if the `constant_boundary` is late-bound but not a scalar.
        TypeError
            if the `constant_boundary` is floating-point but the `data` is
            integer.
        ValueError
            if the `constant_boundary` could not be losslessly converted to the
            `data`'s type.
        """

        with ctx.safeguard(self), ctx.parameter("constant_boundary"):
            constant_boundary = (
                None
                if self._constant_boundary is None
                else late_bound.resolve_ndarray_with_lossless_cast(
                    self._constant_boundary, (), data.dtype
                )
                if isinstance(self._constant_boundary, Parameter)
                else lossless_cast(self._constant_boundary, data.dtype)
            )

        ok: np.ndarray[S, np.dtype[np.bool]] = _ones(
            data.shape, dtype=np.dtype(np.bool)
        )

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
                constant_boundary,
                axis,
            )
            prediction_boundary = _pad_with_boundary(
                prediction,
                self._boundary,
                self._window,
                self._window,
                constant_boundary,
                axis,
            )

            data_windows = sliding_window_view(data_boundary, window, axis=axis)
            prediction_windows = sliding_window_view(
                prediction_boundary, window, axis=axis
            )

            data_monotonic = self._monotonic_sign(data_windows, is_prediction=False)
            prediction_monotonic = self._monotonic_sign(
                prediction_windows, is_prediction=True
            )

            # for monotonic windows, check that the monotonicity matches
            axis_ok = self._monotonic_sign_not_equal(
                data_monotonic, prediction_monotonic
            )

            if self._boundary == BoundaryCondition.valid:
                s = tuple(
                    [slice(None)] * axis
                    + [slice(self._window, -self._window)]
                    + [slice(None)] * (data.ndim - axis - 1)
                )
            else:
                s = tuple([slice(None)] * data.ndim)

            ok[s] &= ~axis_ok.reshape(ok[s].shape)

        return ok

    @override
    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the monotonicity of the `data` is
        preserved.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Data for which the safe intervals should be computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        intervals : IntervalUnion[T, int, int]
            Union of intervals in which the monotonicity is preserved.

        Raises
        ------
        LateBoundParameterResolutionError
            if the `constant_boundary` is late-bound but its late-bound
            parameter is not in `late_bound`.
        ValueError
            if the `constant_boundary` is late-bound but not a scalar.
        TypeError
            if the `constant_boundary` is floating-point but the `data` is
            integer.
        ValueError
            if the `constant_boundary` could not be losslessly converted to the
            `data`'s type.
        """

        with ctx.safeguard(self), ctx.parameter("constant_boundary"):
            constant_boundary = (
                None
                if self._constant_boundary is None
                else late_bound.resolve_ndarray_with_lossless_cast(
                    self._constant_boundary, (), data.dtype
                )
                if isinstance(self._constant_boundary, Parameter)
                else lossless_cast(self._constant_boundary, data.dtype)
            )

        window = 1 + self._window * 2

        (_lt, _gt, _eq, is_weak) = self._monotonicity.value[1]
        nudge = 0 if is_weak else 1

        valid: Interval[T, int] = Interval.full_like(data)

        # track which elements have any monotonicity-based restrictions
        #  imposed upon them
        any_restriction = _zeros(data.shape, dtype=np.dtype(np.bool))

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
                constant_boundary,
                axis,
            )
            data_windows = sliding_window_view(data_boundary, window, axis=axis)
            data_monotonic = self._monotonic_sign(data_windows, is_prediction=False)

            # compute, pointwise, if the element has a decreasing (lt),
            #  increasing (gt), or equality (eq) constraint imposed upon it
            # for the lt and gt variants, compute both a mask for accesses to
            #  elements on the left and to elements on the right, since the
            #  elements at the left and right edge of the window cannot access
            #  one further element to the left / right, respectively
            elem_lt_left, elem_lt_right, elem_eq, elem_gt_left, elem_gt_right = (
                _zeros(data_boundary.shape, dtype=np.dtype(np.bool)),
                _zeros(data_boundary.shape, dtype=np.dtype(np.bool)),
                _zeros(data_boundary.shape, dtype=np.dtype(np.bool)),
                _zeros(data_boundary.shape, dtype=np.dtype(np.bool)),
                _zeros(data_boundary.shape, dtype=np.dtype(np.bool)),
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
                # shape the interval bounds like the data for easier indexing
                valid_lt_lower = valid_lt._lower.reshape(data_boundary.shape)
                valid_lt_upper = valid_lt._upper.reshape(data_boundary.shape)

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
                        valid_lt_lower[s_inner] = from_total_order(
                            _maximum_zero_sign_sensitive(
                                to_total_order(valid_lt_lower[s_inner]),
                                to_total_order(valid_lt_lower[s_boundary]),
                            ),
                            dtype=data.dtype,
                        )
                        valid_lt_upper[s_inner] = from_total_order(
                            _minimum_zero_sign_sensitive(
                                to_total_order(valid_lt_upper[s_inner]),
                                to_total_order(valid_lt_upper[s_boundary]),
                            ),
                            dtype=data.dtype,
                        )

                any_restriction |= elem_lt_left[s] | elem_lt_right[s]
                valid = valid.intersect(
                    Interval(
                        _lower=valid_lt_lower[s].flatten(),
                        _upper=valid_lt_upper[s].flatten(),
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
                # shape the interval bounds like the data for easier indexing
                valid_gt_lower = valid_gt._lower.reshape(data_boundary.shape)
                valid_gt_upper = valid_gt._upper.reshape(data_boundary.shape)

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
                        valid_gt_lower[s_inner] = from_total_order(
                            _maximum_zero_sign_sensitive(
                                to_total_order(valid_gt_lower[s_inner]),
                                to_total_order(valid_gt_lower[s_boundary]),
                            ),
                            dtype=data.dtype,
                        )
                        valid_gt_upper[s_inner] = from_total_order(
                            _minimum_zero_sign_sensitive(
                                to_total_order(valid_gt_upper[s_inner]),
                                to_total_order(valid_gt_upper[s_boundary]),
                            ),
                            dtype=data.dtype,
                        )

                any_restriction |= elem_gt_left[s] | elem_gt_right[s]
                valid = valid.intersect(
                    Interval(
                        _lower=valid_gt_lower[s].flatten(),
                        _upper=valid_gt_upper[s].flatten(),
                    )
                )

        # produce conservative safe intervals by computing the midpoint between
        #  the data and the lower/upper bound
        # for strict monotonicity, the lower bound is nudged up to ensure its
        #  midpoint rounds up while the limiting element's corresponding upper
        #  bound will round down
        lt: np.ndarray[tuple[int], np.dtype[np.unsignedinteger]] = to_total_order(
            valid._lower
        )
        lt = _ensure_array(lt)
        ut: np.ndarray[tuple[int], np.dtype[np.unsignedinteger]] = to_total_order(
            valid._upper
        )
        dt: np.ndarray[tuple[int], np.dtype[np.unsignedinteger]] = to_total_order(
            data.flatten()
        )
        if not is_weak:
            ltp1 = lt + 1
            np.copyto(lt, ltp1, where=np.greater(ltp1, 0), casting="no")

        # Hacker's Delight's algorithm to compute (a + b) / 2:
        #  ((a ^ b) >> 1) + (a & b)
        valid._lower = from_total_order(((lt ^ dt) >> 1) + (lt & dt), data.dtype)  # type: ignore
        valid._upper = from_total_order(((ut ^ dt) >> 1) + (ut & dt), data.dtype)  # type: ignore

        # ensure that non-NaN values remain non-NaN since they can otherwise
        #  invalidate the monotonicity of their window
        valid = valid.intersect(
            Interval.full_like(data).preserve_non_nan(data.flatten())
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

    @override
    def get_config(self) -> dict[str, JSON]:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict[str, JSON]
            Configuration of the safeguard.
        """

        config: dict[str, JSON] = dict(
            kind=type(self).kind,
            monotonicity=self._monotonicity.name,
            window=self._window,
            boundary=self._boundary.name,
            constant_boundary=str(self._constant_boundary)
            if isinstance(self._constant_boundary, Parameter)
            else self._constant_boundary,
            axis=self._axis,
        )

        if self._constant_boundary is None:
            del config["constant_boundary"]

        return config

    def _monotonic_sign(
        self,
        x: np.ndarray[S, np.dtype[T]],
        *,
        is_prediction: bool,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        (lt, gt, eq, _is_weak) = self._monotonicity.value[int(is_prediction)]

        # default to NaN
        monotonic: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.full(
            x.shape[:-1], np.nan
        )

        # use comparison instead of diff to account for uints

        # +1: all(x[i+1] > x[i])
        monotonic[np.all(gt(x[..., 1:], x[..., :-1]), axis=-1)] = +1
        # -1: all(x[i+1] < x[i])
        monotonic[np.all(lt(x[..., 1:], x[..., :-1]), axis=-1)] = -1

        # 0/NaN: all(x[i+1] == x[i])
        monotonic[np.all(x[..., 1:] == x[..., :-1], axis=-1)] = 0 if eq else np.nan

        # NaN values cannot participate in monotonic sequences
        # NaN: any(isnan(x[i]))
        monotonic[np.any(np.isnan(x), axis=-1)] = np.nan

        # return the result in a shape that's broadcastable to x
        return monotonic[..., np.newaxis]

    def _monotonic_sign_not_equal(
        self,
        data_monotonic: np.ndarray[S, np.dtype[T]],
        prediction_monotonic: np.ndarray[S, np.dtype[T]],
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        match self._monotonicity:
            case Monotonicity.strict | Monotonicity.strict_with_consts:
                return ~np.isnan(data_monotonic) & (
                    prediction_monotonic != data_monotonic
                )
            case Monotonicity.strict_to_weak | Monotonicity.weak:
                # having the opposite sign or no sign are both not equal
                return ~np.isnan(data_monotonic) & (
                    (prediction_monotonic == -data_monotonic)
                    | np.isnan(prediction_monotonic)
                )
            case _:
                assert_never(self._monotonicity)
