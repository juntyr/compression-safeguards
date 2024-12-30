import numcodecs
import numcodecs.compat
import numpy as np

from numcodecs.abc import Codec

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

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        return np.all(
            (
                np.abs(self.my_log(data) - self.my_log(decoded))
                <= np.log(1.0 + self._eb_rel)
            )
            | (np.abs(data - decoded) <= self._eb_abs)
            | (self.as_bits(data) == self.as_bits(decoded))
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def encode_correction(
        self, data: np.ndarray, decoded: np.ndarray, *, lossless: Codec
    ) -> bytes:
        log_eb_rel = np.log(1.0 + self._eb_rel)

        data_log = self.my_log(data)
        decoded_log = self.my_log(decoded)

        error_log = decoded_log - data_log
        correction_log = np.round(error_log / (log_eb_rel * 2.0)) * (log_eb_rel * 2.0)

        corrected_log = decoded_log - correction_log
        corrected = self.my_exp(corrected_log)

        corrected = np.where(
            (
                np.abs(data_log - corrected_log)
                <= np.log(1.0 + self._eb_rel)
            )
            | (np.abs(data - corrected) <= self._eb_abs),
            corrected,
            data,
        )

        decoded_bits = self.as_bits(decoded)
        corrected_bits = self.as_bits(corrected, like=decoded)
        correction_bits = decoded_bits - corrected_bits

        correction = lossless.encode(correction_bits)

        return numcodecs.compat.ensure_bytes(correction)

    def apply_correction(
        self, decoded: np.ndarray, correction: bytes, *, lossless: Codec
    ) -> np.ndarray:
        correction = lossless.decode(correction)
        correction = numcodecs.compat.ensure_bytes(correction)

        decoded_bits = self.as_bits(decoded)
        correction_bits = self.as_bits(correction, like=decoded).reshape(decoded.shape)

        return (decoded_bits - correction_bits).view(decoded.dtype)

    def get_config(self):
        return dict(eb_rel=self._eb_rel, eb_abs=self._eb_abs)

    def as_bits(self, a, *, like=None):
        return np.frombuffer(
            a,
            dtype=np.dtype(
                (a if like is None else like)
                .dtype.str.replace("f", "u")
                .replace("i", "u")
            ),
        )
