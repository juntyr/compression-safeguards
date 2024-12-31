import numcodecs
import numcodecs.compat
import numpy as np

from numcodecs.abc import Codec

from . import Guardrail, as_bits, runlength_encode, runlength_decode


class AbsoluteErrorBoundGuardrail(Guardrail):
    __slots__ = ("_eb_abs",)
    _eb_abs: float

    kind = "abs"

    def __init__(self, eb_abs: float):
        assert eb_abs > 0.0, "eb_abs must be positive"
        assert np.isfinite(eb_abs), "eb_abs must be finite"

        self._eb_abs = eb_abs

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
        return (np.abs(data - decoded) <= self._eb_abs) | (
            as_bits(data) == as_bits(decoded)
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def encode_correction(
        self, data: np.ndarray, decoded: np.ndarray, *, lossless: Codec
    ) -> bytes:
        error = decoded - data
        correction = (
            np.round(error / (self._eb_abs * 2.0)) * (self._eb_abs * 2.0)
        ).astype(data.dtype)
        corrected = decoded - correction

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
        return dict(eb_abs=self._eb_abs)
