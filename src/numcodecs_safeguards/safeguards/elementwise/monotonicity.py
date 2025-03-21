"""
Monotonicity-preserving safeguard.
"""

__all__ = ["Monotonicity", "MonotonicityPreservingSafeguard"]

from enum import Enum
from operator import le, lt, ge, gt

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .abc import ElementwiseSafeguard


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

    def _check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
        """
        Check which elements in the `decoded` array preserve the monotonicity
        of the `data` array.

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.

        Returns
        -------
        ok : np.ndarray
            Per-element, `True` if the check succeeded for this element.
        """

        window = 1 + self._window * 2

        needs_correction = np.zeros_like(decoded, dtype=bool)

        flat = np.arange(data.size).reshape(data.shape)
        indices = np.stack(
            np.meshgrid(*[np.arange(a) for a in data.shape], indexing="ij"), axis=-1
        ).reshape(-1, data.ndim)

        for axis, alen in enumerate(data.shape):
            if alen < window:
                continue

            data_windows = sliding_window_view(data, window, axis=axis)
            decoded_windows = sliding_window_view(decoded, window, axis=axis)

            flat_windows = sliding_window_view(flat, window, axis=axis)
            indices_windows = indices[flat_windows]

            data_monotonic = self._monotonic_sign(data_windows, is_decoded=False)

            # for monotonic windows, check that
            #  decoded[i-1] ? decoded[i] ? decoded[i+1]
            # has the correct sign, otherwise mark for correction
            needs_correction[
                indices_windows[..., :-1, :][
                    self._monotonic_sign_not_equal(
                        data_monotonic,
                        self._monotonic_sign_elementwise(
                            decoded_windows, is_decoded=True
                        ),
                    )
                ]
            ] = True
            needs_correction[
                indices_windows[..., 1:, :][
                    self._monotonic_sign_not_equal(
                        data_monotonic,
                        self._monotonic_sign_elementwise(
                            decoded_windows, is_decoded=True
                        ),
                    )
                ]
            ] = True

            # for monotonic windows, check that
            #  data[i-1] ? decoded[i] ? data[i+1]
            # has the correct sign, otherwise mark for correction
            #
            # note: this check is excessively strict but ensures that correcting
            #       to the original data is always possible without affecting
            #       the monotonicity of the corrected output
            needs_correction[
                indices_windows[..., :-1, :][
                    self._monotonic_sign_not_equal(
                        data_monotonic,
                        self._monotonic_sign_elementwise(
                            decoded_windows, data_windows, is_decoded=True
                        ),
                    )
                ]
            ] = True
            needs_correction[
                indices_windows[..., 1:, :][
                    self._monotonic_sign_not_equal(
                        data_monotonic,
                        self._monotonic_sign_elementwise(
                            data_windows, decoded_windows, is_decoded=True
                        ),
                    )
                ]
            ] = True

            # for monotonic windows, check that
            #  decoded[i-1] ? data[i] ? decoded[i+1]
            # has the correct sign, otherwise mark for correction
            #
            # note: this check is excessively strict but ensures that correcting
            #       to the original data is always possible without affecting
            #       the monotonicity of the corrected output
            needs_correction[
                indices_windows[..., :-1, :][
                    self._monotonic_sign_not_equal(
                        data_monotonic,
                        self._monotonic_sign_elementwise(
                            data_windows, decoded_windows, is_decoded=True
                        ),
                    )
                ]
            ] = True
            needs_correction[
                indices_windows[..., 1:, :][
                    self._monotonic_sign_not_equal(
                        data_monotonic,
                        self._monotonic_sign_elementwise(
                            decoded_windows, data_windows, is_decoded=True
                        ),
                    )
                ]
            ] = True

        return ~needs_correction

    def _compute_correction(
        self,
        data: np.ndarray,
        decoded: np.ndarray,
    ) -> np.ndarray:
        return np.where(
            self._check_elementwise(data, decoded),
            decoded,
            data,
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

    def _monotonic_sign_elementwise(
        self,
        left: np.ndarray,
        right: None | np.ndarray = None,
        *,
        is_decoded: bool,
    ) -> np.ndarray:
        right = left if right is None else right

        (lt, gt, eq) = self._monotonicity.value[int(is_decoded)]

        # default to NaN
        monotonic = np.empty(list(left.shape[:-1]) + [left.shape[-1] - 1])
        monotonic.fill(np.nan)

        # use comparison instead of diff to account for uints

        # +1: right[i+1] > left[i]
        monotonic = np.where(gt(right[..., 1:], left[..., :-1]), +1, monotonic)
        # -1: right[i+1] < left[i]
        monotonic = np.where(lt(right[..., 1:], left[..., :-1]), -1, monotonic)

        # 0/NaN: right[i+1] == left[i]
        monotonic = np.where(
            right[..., 1:] == left[..., :-1], 0 if eq else np.nan, monotonic
        )

        # non-finite values cannot participate in monotonic sequences
        # NaN: !(isfinite(right[i+1]) && isfinite(lift[i]))
        monotonic = np.where(
            np.isfinite(right[..., 1:]) & np.isfinite(left[..., :-1]), monotonic, np.nan
        )

        return monotonic

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
            case Monotonicity.strict_to_weak | Monotonicity.weak:
                return np.where(
                    np.isfinite(data_monotonic),
                    # having the opposite sign or no sign are both not equal
                    (decoded_monotonic == -data_monotonic)
                    | np.isnan(decoded_monotonic),
                    False,
                )
