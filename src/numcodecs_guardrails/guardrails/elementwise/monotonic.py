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

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
        window = 1 + self._window * 2

        needs_correction = np.zeros_like(decoded, dtype=bool)

        for axis, alen in enumerate(data.shape):
            if alen < window:
                continue

            data_windows = sliding_window_view(data, window, axis=axis)
            decoded_windows = sliding_window_view(decoded, window, axis=axis)

            data_monotonic = (
                np.all(np.diff(data_windows, axis=-1) > 0) * 1
                - np.all(np.diff(data_windows, axis=-1) < 0) * 1
            )
            decoded_monotonic = (
                np.all(np.diff(decoded_windows, axis=-1) > 0) * 1
                - np.all(np.diff(decoded_windows, axis=-1) < 0) * 1
            )

            if np.all(
                np.where(data_monotonic == 0, True, data_monotonic == decoded_monotonic)
            ):
                continue

            # TODO
            needs_correction |= True

        return ~needs_correction

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def compute_correction(
        self,
        data: np.ndarray,
        decoded: np.ndarray,
    ) -> np.ndarray:
        # TODO
        return data

    def get_config(self) -> dict:
        return dict(kind=type(self).kind, window=self._window)
