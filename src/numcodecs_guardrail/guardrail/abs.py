import numcodecs
import numcodecs.compat
import numpy as np

from numcodecs.abc import Codec

from . import Guardrail


class AbsoluteErrorBoundGuardrail(Guardrail):
    __slots__ = ("_eb_abs",)
    _eb_abs: float

    kind = "abs"

    def __init__(self, eb_abs: float):
        assert eb_abs > 0.0, "eb_abs must be positive"

        self._eb_abs = eb_abs

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        return np.all(
            (np.abs(data - decoded) <= self._eb_abs)
            | (self.as_bits(data) == self.as_bits(decoded))
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def encode_correction(
        self, data: np.ndarray, decoded: np.ndarray, *, lossless: Codec
    ) -> bytes:
        error = decoded - data
        correction = np.round(error / (self._eb_abs * 2.0)) * (self._eb_abs * 2.0)
        corrected = decoded - correction

        corrected = np.where(
            np.abs(data - corrected) <= self._eb_abs,
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
        return dict(eb_abs=self._eb_abs)

    def as_bits(self, a, *, like=None):
        return np.frombuffer(
            a,
            dtype=np.dtype(
                (a if like is None else like)
                .dtype.str.replace("f", "u")
                .replace("i", "u")
            ),
        )
