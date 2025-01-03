__all__ = ["MonotonicGuardrail"]

import numpy as np

from numpy.lib.stride_tricks import sliding_window_view

from . import ElementwiseGuardrail


class MonotonicGuardrail(ElementwiseGuardrail):
    __slots__ = "_window"
    _window: int

    kind = "monotonic"
    _priority = 1

    def __init__(self, window: int):
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
                    data_monotonic == 0, False, data_monotonic != decoded_monotonic
                )
            ):
                return False

        return True

    def check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
        window = 1 + self._window * 2

        needs_correction = np.zeros_like(decoded, dtype=bool)

        for axis, alen in enumerate(data.shape):
            if alen < window:
                continue

            data_windows = sliding_window_view(data, window, axis=axis)
            decoded_windows = sliding_window_view(decoded, window, axis=axis)
            correction_windows = sliding_window_view(
                needs_correction, window, axis=axis, writeable=True
            )

            data_monotonic = self._strictly_monotonic_sign(data_windows)

            # for strictly monotonic windows, check that
            #  decoded[i-1] ? decoded[i] ? decoded[i+1]
            # has the correct sign, otherwise mark for correction
            correction_windows[..., :-1] |= (
                (
                    (decoded_windows[..., 1:] > decoded_windows[..., :-1]) * 1
                    - (decoded_windows[..., 1:] < decoded_windows[..., :-1]) * 1
                )
                != data_monotonic
            ) & (data_monotonic != 0)
            correction_windows[..., 1:] |= (
                (
                    (decoded_windows[..., 1:] > decoded_windows[..., :-1]) * 1
                    - (decoded_windows[..., 1:] < decoded_windows[..., :-1]) * 1
                )
                != data_monotonic
            ) & (data_monotonic != 0)

            # for strictly monotonic windows, check that
            #  data[i-1] ? decoded[i] ? data[i+1]
            # has the correct sign, otherwise mark for correction
            #
            # note: this check is excessively strict but ensures that correcting
            #       to the original data is always possible without affecting
            #       the monotonicity of the corrected output
            correction_windows[..., :-1] |= (
                (
                    (data_windows[..., 1:] > decoded_windows[..., :-1]) * 1
                    - (data_windows[..., 1:] < decoded_windows[..., :-1]) * 1
                )
                != data_monotonic
            ) & (data_monotonic != 0)
            correction_windows[..., 1:] |= (
                (
                    (decoded_windows[..., 1:] > data_windows[..., :-1]) * 1
                    - (decoded_windows[..., 1:] < data_windows[..., :-1]) * 1
                )
                != data_monotonic
            ) & (data_monotonic != 0)

        return ~needs_correction

    def compute_correction(
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
        return dict(kind=type(self).kind, window=self._window)

    def _strictly_monotonic_sign(x: np.ndarray) -> np.ndarray:
        # use comparison instead of diff to account for uints
        monotonic = (
            np.all(x[..., 1:] > x[..., :-1], axis=-1) * 1
            - np.all(x[..., 1:] < x[..., :-1], axis=-1) * 1
        )
        # non-finite values cannot participate in monotonic sequences
        monotonic *= np.all(np.isfinite(x), axis=-1)

        # return the result in a shape that's broadcastable to x
        return monotonic[..., np.newaxis]
