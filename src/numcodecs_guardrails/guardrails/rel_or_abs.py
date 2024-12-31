import numcodecs
import numcodecs.compat
import numpy as np

from numcodecs.abc import Codec

from . import Guardrail, as_bits, runlength_decode, runlength_encode


class RelativeOrAbsoluteErrorBoundGuardrail(Guardrail):
    __slots__ = ("_eb_rel", "_eb_abs")
    _eb_rel: float
    _eb_abs: float

    kind = "rel_or_abs"

    def __init__(self, eb_rel: float, eb_abs: float):
        assert eb_rel > 0.0, "eb_rel must be positive"
        assert np.isfinite(eb_rel), "eb_rel must be finite"
        assert eb_abs > 0.0, "eb_abs must be positive"
        assert np.isfinite(eb_abs), "eb_abs must be finite"

        self._eb_rel = eb_rel
        self._eb_abs = eb_abs

    def my_log(self, x: np.ndarray) -> np.ndarray:
        a = 0.5 * self._eb_abs / (1.0 + self._eb_rel)

        return np.sign(x) * np.log(np.maximum(np.abs(x), a) / a)

    def my_exp(self, x: np.ndarray) -> np.ndarray:
        a = 0.5 * self._eb_abs / (1.0 + self._eb_rel)

        return np.sign(x) * np.exp(np.abs(x)) * a

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
        return (
            (
                np.abs(self.my_log(data) - self.my_log(decoded))
                <= np.log(1.0 + self._eb_rel)
            )
            | (np.abs(data - decoded) <= self._eb_abs)
            | (as_bits(data) == as_bits(decoded))
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
        corrected = self.my_exp(corrected_log).astype(data.dtype)

        corrected = np.where(
            self.check_elementwise(data, corrected),
            corrected,
            data,
        )

        decoded_bits = as_bits(decoded)
        corrected_bits = as_bits(corrected, like=decoded)
        correction_bits = decoded_bits - corrected_bits

        correction = runlength_encode(correction_bits)
        correction = lossless.encode(correction)

        return numcodecs.compat.ensure_bytes(correction)

    def apply_correction(
        self, decoded: np.ndarray, correction: bytes, *, lossless: Codec
    ) -> np.ndarray:
        decoded_bits = as_bits(decoded)

        correction = lossless.decode(correction)
        correction = numcodecs.compat.ensure_bytes(correction)

        correction_bits = runlength_decode(correction, like=decoded_bits)

        return (decoded_bits - correction_bits).view(decoded.dtype)

    def get_config(self):
        return dict(eb_rel=self._eb_rel, eb_abs=self._eb_abs)
