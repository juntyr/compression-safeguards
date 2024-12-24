import numcodecs
import numcodecs.abc
import numcodecs.compat
import numcodecs.registry
import numcodecs.zlib
import numpy as np

from . import Guardrail


class RelativeOrAbsoluteErrorBoundGuardrail(Guardrail):
    __slots__ = ("_eb_rel", "_eb_abs")
    _eb_rel: float
    _eb_abs: float

    kind = "rel_or_abs"

    def __init__(self, eb_rel: float, eb_abs: float):
        assert eb_rel > 0.0, "eb_rel must be positive"
        assert eb_abs > 0.0, "eb_abs must be positive"

        self._eb_rel = eb_rel
        self._eb_abs = eb_abs

    def my_log(self, x: np.ndarray) -> np.ndarray:
        a = 0.5 * self._eb_abs / (1.0 + self._eb_rel)

        return np.sign(x) * np.log(np.maximum(np.abs(x), a) / a)

    def my_exp(self, x: np.ndarray) -> np.ndarray:
        a = 0.5 * self._eb_abs / (1.0 + self._eb_rel)

        return np.sign(x) * np.exp(np.abs(x)) * a

    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        return np.all(
            (
                np.abs(self.my_log(data) - self.my_log(decoded))
                <= np.log(1.0 + self._eb_rel)
            )
            | (np.abs(data - decoded) <= self._eb_abs)
        )

    def encode_correction(self, data: np.ndarray, decoded: np.ndarray) -> bytes:
        log_eb_rel = np.log(1.0 + self._eb_rel)

        data_log = self.my_log(data)
        decoded_log = self.my_log(decoded)

        error_log = decoded_log - data_log
        correction_log = np.round(error_log / (log_eb_rel * 2.0))

        correction = self._lossless.encode(correction_log)

        return numcodecs.compat.ensure_bytes(correction)

    def apply_correction(self, decoded: np.ndarray, correction: bytes) -> np.ndarray:
        log_eb_rel = np.log(1.0 + self._eb_rel)

        correction = self._lossless.decode(correction)
        correction = numcodecs.compat.ensure_bytes(correction)
        correction = np.frombuffer(correction, np.int64).reshape(decoded.shape)
        correction_log = correction.astype(decoded.dtype) * (log_eb_rel * 2.0)

        decoded_log = self.my_log(decoded)
        corrected_log = decoded_log - correction_log

        return self.my_exp(corrected_log)

    def get_config(self):
        return dict(eb_rel=self._eb_rel, eb_abs=self._eb_abs)
