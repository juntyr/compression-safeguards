"""
Implementation of the [`SafeguardsQuantizer`][numcodecs_safeguards.quantizer.SafeguardsQuantizer], which quantizes the correction needed to satisfy a set of safeguards.
"""

__all__ = ["SafeguardsQuantizer"]

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

from .cast import as_bits
from .safeguards import Safeguards
from .safeguards.abc import Safeguard
from .safeguards.elementwise.abc import ElementwiseSafeguard

T = TypeVar("T", bound=np.dtype)
""" Any numpy [`dtype`][numpy.dtype] type variable. """
C = TypeVar("C", bound=np.dtype)
""" The numpy [`dtype`][numpy.dtype] type variable for corrections. """
S = TypeVar("S", bound=tuple[int, ...])
""" Any array shape. """


class SafeguardsQuantizer:
    """
    A quantizer which computes the correction needed to satisfy a set of `safeguards`.

    Parameters
    ----------
    safeguards : Sequence[dict | Safeguard]
        The safeguards that will be applied to the codec. They can either be
        passed as a safeguard configuration [`dict`][dict] or an already
        initialized
        [`Safeguard`][numcodecs_safeguards.safeguards.abc.Safeguard].

        Please refer to
        [`Safeguards`][numcodecs_safeguards.safeguards.Safeguards]
        for an enumeration of all supported safeguards.
    _version : ...
        Internal, do not provide this paramter explicitly.
    """

    __slots__ = ("_elementwise_safeguards",)
    _elementwise_safeguards: tuple[ElementwiseSafeguard, ...]

    def __init__(
        self,
        *,
        safeguards: Sequence[dict | Safeguard],
        _version: None | str = None,
    ):
        if _version is not None:
            assert _version == _FORMAT_VERSION

        safeguards = [
            safeguard
            if isinstance(safeguard, Safeguard)
            else Safeguards[safeguard["kind"]].value(
                **{p: v for p, v in safeguard.items() if p != "kind"}
            )
            for safeguard in safeguards
        ]

        self._elementwise_safeguards = tuple(
            safeguard
            for safeguard in safeguards
            if isinstance(safeguard, ElementwiseSafeguard)
        )
        safeguards = [
            safeguard
            for safeguard in safeguards
            if not isinstance(safeguard, ElementwiseSafeguard)
        ]

        assert len(safeguards) == 0, f"unsupported safeguards {safeguards:!r}"

    @property
    def safeguards(self) -> tuple[Safeguard, ...]:
        """
        The set of safeguards that this quantizer has been configured to
        uphold.
        """

        return self._elementwise_safeguards

    @property
    def version(self) -> str:
        """
        The version of the format of the correction computed by the
        [`quantize`][numcodecs_safeguards.quantizer.SafeguardsQuantizer.quantize]
        method.

        The quantizer can only
        [`recover`][numcodecs_safeguards.quantizer.SafeguardsQuantizer.recover]
        quantized corrections with the matching version.
        """

        return _FORMAT_VERSION

    def quantize(
        self, data: np.ndarray[S, T], prediction: np.ndarray[S, T]
    ) -> np.ndarray[S, C]:
        """
        Quantize the correction required to make the `prediction` array satisfy the safeguards relative to the `data` array.

        Parameters
        ----------
        data : np.ndarray[S, T]
            The data array, relative to which the safeguards are enforced.
        prediction : np.ndarray[S, T]
            The prediction array for which the correction is computed.

        Returns
        -------
        correction : np.ndarray[S, C]
            The correction array.
        """

        assert data.dtype in _SUPPORTED_DTYPES, (
            f"can only quantize arrays of dtype {', '.join(d.str for d in _SUPPORTED_DTYPES)}"
        )

        all_ok = True
        for safeguard in self._elementwise_safeguards:
            if not safeguard.check(data, prediction):
                all_ok = False
                break

        if all_ok:
            return np.zeros_like(as_bits(data))

        all_intervals = []
        for safeguard in self._elementwise_safeguards:
            intervals = safeguard.compute_safe_intervals(data)
            assert np.all(intervals.contains(data)), (
                f"elementwise safeguard {safeguard!r}'s intervals must contain the original data"
            )
            all_intervals.append(intervals)

        combined_intervals = all_intervals[0]
        for intervals in all_intervals[1:]:
            combined_intervals = combined_intervals.intersect(intervals)
        correction = combined_intervals.encode(prediction)

        for safeguard, intervals in zip(self._elementwise_safeguards, all_intervals):
            assert np.all(intervals.contains(correction)), (
                f"{safeguard!r} interval does not contain the correction {correction!r}"
            )
            assert safeguard.check(data, correction), (
                f"{safeguard!r} check fails after correction {correction!r}"
            )

        prediction_bits = as_bits(prediction)
        correction_bits = as_bits(correction)

        return prediction_bits - correction_bits

    def recover(
        self, prediction: np.ndarray[S, T], quantized: np.ndarray[S, C]
    ) -> np.ndarray[S, T]:
        """
        Recover the corrected array from the `prediction` and its quantized `correction`.

        Parameters
        ----------
        prediction : np.ndarray[S, T]
            The prediction array for which the correction has been computed.
        quantized : quantized: np.ndarray[S, C]
            The quantized correction array.

        Returns
        -------
        corrected : np.ndarray[T, C]
            The corrected array, which satisfies the safeguards.
        """

        prediction_bits = as_bits(prediction)
        quantized_bits = as_bits(quantized)

        recovered = prediction_bits - quantized_bits

        return recovered.view(prediction.dtype)

    def get_config(self) -> dict:
        """
        Returns the configuration of the quantizer with safeguards.

        Returns
        -------
        config : dict
            Configuration of the quantizer with safeguards.
        """

        return dict(
            _version=self.version,
            safeguards=[safeguard.get_config() for safeguard in self.safeguards],
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(safeguards={list(self.safeguards)!r})"


_FORMAT_VERSION: str = "0.1.x"


_SUPPORTED_DTYPES: set[np.dtype] = {
    np.dtype("int8"),
    np.dtype("int16"),
    np.dtype("int32"),
    np.dtype("int64"),
    np.dtype("uint8"),
    np.dtype("uint16"),
    np.dtype("uint32"),
    np.dtype("uint64"),
    np.dtype("float32"),
    np.dtype("float64"),
}
