import numcodecs
import numcodecs.abc
import numcodecs.compat
import numcodecs.registry
import numcodecs.zlib
import numpy as np

from . import Guardrail


class AbsoluteErrorBoundGuardrail(Guardrail):
    __slots__ = "_eb_abs"
    _eb_abs: float

    kind = "abs"

    def __init__(self, eb_abs: float):
        assert eb_abs is not None, "eb_abs must not be None"
        assert eb_abs > 0.0, "eb_abs must be positive"

        self._eb_abs = eb_abs

    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        return np.amax(np.abs(data - decoded)) <= self._eb_abs

    def encode_correction(self, data: np.ndarray, decoded: np.ndarray) -> bytes:
        error = decoded - data
        correction = np.round(error / (self._eb_abs * 2.0)).astype(np.int64)

        correction = self._lossless.encode(correction)

        return numcodecs.compat.ensure_bytes(correction)

    def apply_correction(self, decoded: np.ndarray, correction: bytes) -> np.ndarray:
        correction = self._lossless.decode(correction)
        correction = numcodecs.compat.ensure_bytes(correction)
        correction = np.frombuffer(correction, np.int64).reshape(decoded.shape)
        correction = correction.astype(decoded.dtype) * (self._eb_abs * 2.0)

        return decoded + correction

    def get_config(self):
        return dict(eb_abs=self._eb_abs)
