"""
Monotonicity-preserving safeguard.
"""

__all__ = ["Monotonicity", "MonotonicityPreservingSafeguard"]

from enum import Enum
from operator import le, lt, ge, gt

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .abc import ElementwiseSafeguard
from ...intervals import (
    IntervalUnion,
    Interval,
    _to_total_order,
    _from_total_order,
)


_STRICT = ((lt, gt, False),) * 2
_STRICT_WITH_CONSTS = ((lt, gt, True),) * 2
_STRICT_TO_WEAK = ((lt, gt, False), (le, ge, True))
_WEAK = ((le, ge, False), (le, ge, True))


class Monotonicity(Enum):
    """
    Different levels of monotonicity that can be enforced by the
    [`MonotonicityPreservingSafeguard`][numcodecs_safeguards.safeguards.elementwise.monotonicity.MonotonicityPreservingSafeguard].
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

    # strict_to_weak = _STRICT_TO_WEAK
    # """
    # Strictly increasing/decreasing sequences in the input array are guaranteed
    # to be *weakly* increasing/decreasing (or constant) in the decoded array.

    # Sequences that are not strictly increasing/decreasing or contain non-finite
    # values are not affected.
    # """

    # weak = _WEAK
    # """
    # Weakly increasing/decreasing (but not constant) sequences in the input
    # array are guaranteed to be weakly increasing/decreasing (or constant) in
    # the decoded array.

    # Sequences that are not weakly increasing/decreasing or are constant or
    # contain non-finite values are not affected.
    # """


class MonotonicityPreservingSafeguard(ElementwiseSafeguard):
    r"""
    The `MonotonicityPreservingSafeguard` guarantees that sequences that are
    monotonic in the input are guaranteed to be monotonic in the decompressed
    output.

    Monotonic sequences are detected using per-axis moving windows with a
    constant symmetric size of $(1 + window \cdot 2)$. Typically, the window
    size should be chosen to be large enough to ignore noise, i.e. $>1$, but
    small enough to capture details.

    The safeguard supports enforcing four levels of
    [`Monotonicity`][numcodecs_safeguards.safeguards.elementwise.monotonicity.Monotonicity]:
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
    """

    __slots__ = "_window"
    _window: int
    _monotonicity: Monotonicity

    kind = "monotonicity"

    def __init__(self, monotonicity: str | Monotonicity, window: int):
        self._monotonicity = (
            monotonicity
            if isinstance(monotonicity, Monotonicity)
            else Monotonicity[monotonicity]
        )

        assert window > 0, "window size must be positive"
        self._window = window

    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        """
        Check if monotonic sequences in the `data` array are preserved in the
        `decoded` array.

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.

        Returns
        -------
        ok : bool
            `True` if the check succeeded.
        """

        window = 1 + self._window * 2

        for axis, alen in enumerate(data.shape):
            if alen < window:
                continue

            data_windows = sliding_window_view(data, window, axis=axis)
            decoded_windows = sliding_window_view(decoded, window, axis=axis)

            data_monotonic = self._monotonic_sign(data_windows, is_decoded=False)
            decoded_monotonic = self._monotonic_sign(decoded_windows, is_decoded=True)

            # for monotonic windows, check that the monotonicity matches
            if np.any(
                self._monotonic_sign_not_equal(data_monotonic, decoded_monotonic)
            ):
                return False

        return True

    def compute_safe_intervals(self, data: np.ndarray) -> IntervalUnion:
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

        valid = Interval.full_like(data)

        for axis, alen in enumerate(data.shape):
            if alen < window:
                continue

            axis_valid = Interval.full_like(data)
            lower = axis_valid._lower.reshape(data.shape)
            upper = axis_valid._upper.reshape(data.shape)

            data_windows = sliding_window_view(data, window, axis=axis)
            data_monotonic = self._monotonic_sign(data_windows, is_decoded=False)

            print(data_windows.shape, data_monotonic.shape)

            elem_lt, elem_eq, elem_gt = (
                np.zeros_like(data, dtype=np.bool),
                np.zeros_like(data, dtype=np.bool),
                np.zeros_like(data, dtype=np.bool),
            )
            for w in range(window):
                print(data.shape, window, axis, w, tuple(
                        [slice(None)] * axis
                        + [slice(w, None if (w + 1) == window else -window + w + 1)]
                        + [slice(None)] * (len(data.shape) - axis - 1)
                    ))
                # problem here: data_monotonic has reduced axis dimensions
                elem_lt[
                    tuple(
                        [slice(None)] * axis
                        + [slice(w, None if (w + 1) == window else -window + w + 1)]
                        + [slice(None)] * (len(data.shape) - axis - 1)
                    )
                ] = data_monotonic[..., w] == -1
                elem_eq[
                    tuple(
                        [slice(None)] * axis
                        + [slice(w, None if (w + 1) == window else -window + w + 1)]
                        + [slice(None)] * (len(data.shape) - axis - 1)
                    )
                ] = data_monotonic[..., w] == 0
                elem_gt[
                    tuple(
                        [slice(None)] * axis
                        + [slice(w, None if (w + 1) == window else -window + w + 1)]
                        + [slice(None)] * (len(data.shape) - axis - 1)
                    )
                ] = data_monotonic[..., w] == +1

            elem_monotonic = np.where(
                elem_eq, 0, np.where(elem_lt, -1, np.where(elem_gt, +1, np.nan))
            )

            lower[elem_monotonic == 0] = data[elem_monotonic == 0]
            upper[elem_monotonic == 0] = data[elem_monotonic == 0]

            mask = np.zeros(alen, dtype=np.bool)
            mask[-1] = False
            mask = mask.reshape(
                [1] * axis + [alen] + [1] * (len(data.shape) - axis - 1)
            )

            lower[(elem_monotonic == -1) & mask] = _from_total_order(
                _to_total_order(
                    np.roll(data, -1, axis=axis)[(elem_monotonic == -1) & mask]
                )
                + 1,
                dtype=data.dtype,
            )
            upper[(elem_monotonic == +1) & mask] = _from_total_order(
                _to_total_order(
                    np.roll(data, -1, axis=axis)[(elem_monotonic == +1) & mask]
                )
                - 1,
                dtype=data.dtype,
            )

            mask = np.zeros(alen, dtype=np.bool)
            mask[0] = False
            mask = mask.reshape(
                [1] * axis + [alen] + [1] * (len(data.shape) - axis - 1)
            )

            upper[(elem_monotonic == -1) & mask] = _from_total_order(
                _to_total_order(
                    np.roll(data, +1, axis=axis)[(elem_monotonic == -1) & mask]
                )
                - 1,
                dtype=data.dtype,
            )
            lower[(elem_monotonic == +1) & mask] = _from_total_order(
                _to_total_order(
                    np.roll(data, +1, axis=axis)[(elem_monotonic == +1) & mask]
                )
                + 1,
                dtype=data.dtype,
            )

            valid = valid.intersect(axis_valid)

        lt, ut, dt = (
            _to_total_order(valid._lower),
            _to_total_order(valid._upper),
            _to_total_order(data.flatten()),
        )
        lt = np.where((lt + 1) > 0, lt + 1, lt)

        # Hacker's Delight's algorithm to compute (a + b) / 2:
        #  ((a ^ b) >> 1) + (a & b)
        valid._lower = ((lt ^ dt) >> 1) + (lt & dt)
        valid._upper = ((ut ^ dt) >> 1) + (ut & dt)

        return valid.into_union()

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
            monotonicity=self._monotonicity.name,
            window=self._window,
        )

    def _monotonic_sign(
        self,
        x: np.ndarray,
        *,
        is_decoded: bool,
    ) -> np.ndarray:
        (lt, gt, eq) = self._monotonicity.value[int(is_decoded)]

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
        monotonic = np.where(np.all(np.isfinite(x), axis=-1), monotonic, np.nan)

        # return the result in a shape that's broadcastable to x
        return monotonic[..., np.newaxis]

    def _monotonic_sign_not_equal(
        self, data_monotonic: np.ndarray, decoded_monotonic: np.ndarray
    ) -> np.ndarray:
        match self._monotonicity:
            case Monotonicity.strict | Monotonicity.strict_with_consts:
                return np.where(
                    np.isfinite(data_monotonic),
                    decoded_monotonic != data_monotonic,
                    False,
                )
            # case Monotonicity.strict_to_weak | Monotonicity.weak:
            #     return np.where(
            #         np.isfinite(data_monotonic),
            #         # having the opposite sign or no sign are both not equal
            #         (decoded_monotonic == -data_monotonic)
            #         | np.isnan(decoded_monotonic),
            #         False,
            #     )
