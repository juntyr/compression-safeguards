__all__ = ["RelativeOrAbsoluteErrorBoundGuardrail"]

import numpy as np

from . import ElementwiseGuardrail, _as_bits


class RelativeOrAbsoluteErrorBoundGuardrail(ElementwiseGuardrail):
    __slots__ = ("_eb_rel", "_eb_abs")
    _eb_rel: float
    _eb_abs: float

    kind = "rel_or_abs"
    _priority = 0

    def __init__(self, eb_rel: float, eb_abs: float):
        assert eb_rel > 0.0, "eb_rel must be positive"
        assert np.isfinite(eb_rel), "eb_rel must be finite"
        assert eb_abs > 0.0, "eb_abs must be positive"
        assert np.isfinite(eb_abs), "eb_abs must be finite"

        self._eb_rel = eb_rel
        self._eb_abs = eb_abs

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
        return (
            (np.abs(self._log(data) - self._log(decoded)) <= np.log(1.0 + self._eb_rel))
            | (np.abs(data - decoded) <= self._eb_abs)
            | (_as_bits(data) == _as_bits(decoded))
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def compute_correction(
        self,
        data: np.ndarray,
        decoded: np.ndarray,
    ) -> np.ndarray:
        log_eb_rel = np.log(1.0 + self._eb_rel)

        data_log = self._log(data)
        decoded_log = self._log(decoded)

        error_log = decoded_log - data_log
        correction_log = np.round(error_log / (log_eb_rel * 2.0)) * (log_eb_rel * 2.0)

        corrected_log = decoded_log - correction_log
        corrected = self._exp(corrected_log).astype(data.dtype)

        return np.where(
            self.check_elementwise(data, corrected),
            corrected,
            data,
        )

    def get_config(self) -> dict:
        return dict(kind=type(self).kind, eb_rel=self._eb_rel, eb_abs=self._eb_abs)

    def _log(self, x: np.ndarray) -> np.ndarray:
        a = 0.5 * self._eb_abs / (1.0 + self._eb_rel)

        return np.sign(x) * np.log(np.maximum(np.abs(x), a) / a)

    def _exp(self, x: np.ndarray) -> np.ndarray:
        a = 0.5 * self._eb_abs / (1.0 + self._eb_rel)

        return np.sign(x) * np.exp(np.abs(x)) * a
