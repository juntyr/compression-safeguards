__all__ = ["MonotonicGuardrail"]

from enum import Enum
from operator import lt, gt  # , le, ge
from typing import Optional

import numpy as np

from numpy.lib.stride_tricks import sliding_window_view

from . import ElementwiseGuardrail


class Monotonicity(Enum):
    strict = (lt, gt)
    strict_with_consts = (lt, gt)
    # strict_to_weak = ()
    # weak = (le, ge)


class MonotonicGuardrail(ElementwiseGuardrail):
    __slots__ = "_window"
    _window: int
    _monotonicity: Monotonicity

    kind = "monotonic"
    _priority = 1

    def __init__(self, monotonicity: str | Monotonicity, window: int):
        self._monotonicity = (
            monotonicity
            if isinstance(monotonicity, Monotonicity)
            else Monotonicity[monotonicity]
        )

        assert window > 0, "window size must be positive"
        self._window = window

    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        window = 1 + self._window * 2

        for axis, alen in enumerate(data.shape):
            if alen < window:
                continue

            data_windows = sliding_window_view(data, window, axis=axis)
            decoded_windows = sliding_window_view(decoded, window, axis=axis)

            data_monotonic = self._strictly_monotonic_sign(data_windows)
            decoded_monotonic = self._strictly_monotonic_sign(decoded_windows)

            # for strictly monotonic windows, check that the monotonicity
            # matches
            if np.any(
                np.where(
                    np.isfinite(data_monotonic),
                    data_monotonic != decoded_monotonic,
                    False,
                )
            ):
                return False

        return True

    def check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
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

            data_monotonic = self._strictly_monotonic_sign(data_windows)

            # for strictly monotonic windows, check that
            #  decoded[i-1] ? decoded[i] ? decoded[i+1]
            # has the correct sign, otherwise mark for correction
            needs_correction[
                indices_windows[..., :-1, :][
                    (
                        self._strictly_monotonic_sign_elementwise(decoded_windows)
                        != data_monotonic
                    )
                    & np.isfinite(data_monotonic)
                ]
            ] = True
            needs_correction[
                indices_windows[..., 1:, :][
                    (
                        self._strictly_monotonic_sign_elementwise(decoded_windows)
                        != data_monotonic
                    )
                    & np.isfinite(data_monotonic)
                ]
            ] = True

            # for strictly monotonic windows, check that
            #  data[i-1] ? decoded[i] ? data[i+1]
            # has the correct sign, otherwise mark for correction
            #
            # note: this check is excessively strict but ensures that correcting
            #       to the original data is always possible without affecting
            #       the monotonicity of the corrected output
            needs_correction[
                indices_windows[..., :-1, :][
                    (
                        self._strictly_monotonic_sign_elementwise(
                            decoded_windows, data_windows
                        )
                        != data_monotonic
                    )
                    & np.isfinite(data_monotonic)
                ]
            ] = True
            needs_correction[
                indices_windows[..., 1:, :][
                    (
                        self._strictly_monotonic_sign_elementwise(
                            data_windows, decoded_windows
                        )
                        != data_monotonic
                    )
                    & np.isfinite(data_monotonic)
                ]
            ] = True

        return ~needs_correction

    def _compute_correction(
        self,
        data: np.ndarray,
        decoded: np.ndarray,
    ) -> np.ndarray:
        return np.where(
            self.check_elementwise(data, decoded),
            decoded,
            data,
        )

    def get_config(self) -> dict:
        """
        Returns the configuration of the guardrail.

        Returns
        -------
        config : dict
            Configuration of the guardrail.
        """

        return dict(
            kind=type(self).kind,
            monotonicity=self._monotonicity.name,
            window=self._window,
        )

    def _strictly_monotonic_sign(
        self, x: np.ndarray, *, monotonicity: Optional[Monotonicity] = None
    ) -> np.ndarray:
        monotonicity = self._monotonicity if monotonicity is None else monotonicity
        (lt, gt) = monotonicity.value

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

        # 0: all(x[i+1] == x[i])
        if monotonicity in (Monotonicity.strict_with_consts,):  # Monotonicity.weak):
            monotonic = np.where(
                np.all(x[..., 1:] == x[..., :-1], axis=-1), 0, monotonic
            )

        # non-finite values cannot participate in monotonic sequences
        # NaN: any(!isfinite(x[i]))
        monotonic = np.where(np.all(np.isfinite(x), axis=-1), monotonic, np.nan)

        # return the result in a shape that's broadcastable to x
        return monotonic[..., np.newaxis]

    def _strictly_monotonic_sign_elementwise(
        self,
        left: np.ndarray,
        right: Optional[np.ndarray] = None,
        *,
        monotonicity: Optional[Monotonicity] = None,
    ) -> np.ndarray:
        right = left if right is None else right

        monotonicity = self._monotonicity if monotonicity is None else monotonicity
        (lt, gt) = monotonicity.value

        # default to NaN
        monotonic = np.empty(list(left.shape[:-1]) + [left.shape[-1] - 1])
        monotonic.fill(np.nan)

        # use comparison instead of diff to account for uints

        # +1: right[i+1] > left[i]
        monotonic = np.where(gt(right[..., 1:], left[..., :-1]), +1, monotonic)
        # -1: right[i+1] < left[i]
        monotonic = np.where(lt(right[..., 1:], left[..., :-1]), -1, monotonic)

        # 0: right[i+1] == left[i]
        if monotonicity in (Monotonicity.strict_with_consts,):  # Monotonicity.weak):
            monotonic = np.where(right[..., 1:] == left[..., :-1], 0, monotonic)

        # non-finite values cannot participate in monotonic sequences
        # NaN: !(isfinite(right[i+1]) && isfinite(lift[i]))
        monotonic = np.where(
            np.isfinite(right[..., 1:]) & np.isfinite(left[..., :-1]), monotonic, np.nan
        )

        return monotonic
