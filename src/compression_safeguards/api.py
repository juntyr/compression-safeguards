"""
Implementation of the [`Safeguards`][compression_safeguards.api.Safeguards], which compute the correction needed to satisfy a set of [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard]s.
"""

__all__ = ["Safeguards"]

from collections.abc import Collection, Mapping, Set

import numpy as np
from typing_extensions import Self  # MSPV 3.11

from .safeguards import SafeguardKind
from .safeguards.abc import Safeguard
from .safeguards.pointwise.abc import PointwiseSafeguard
from .safeguards.stencil.abc import StencilSafeguard
from .utils.bindings import Bindings, Parameter, Value
from .utils.cast import as_bits
from .utils.typing import C, S, T


class Safeguards:
    """
    Collection of [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard]s.

    Parameters
    ----------
    safeguards : Collection[dict | Safeguard]
        The safeguards that will be applied. They can either be passed as a
        safeguard configuration [`dict`][dict] or an already initialized
        [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard].

        Please refer to the
        [`SafeguardKind`][compression_safeguards.safeguards.SafeguardKind]
        for an enumeration of all supported safeguards.
    _version : ...
        The safeguards' version. Do not provide this parameter explicitly.
    """

    __slots__ = ("_pointwise_safeguards", "_stencil_safeguards")
    _pointwise_safeguards: tuple[PointwiseSafeguard, ...]
    _stencil_safeguards: tuple[StencilSafeguard, ...]

    def __init__(
        self,
        *,
        safeguards: Collection[dict | Safeguard],
        _version: None | str = None,
    ) -> None:
        if _version is not None:
            assert _version == _FORMAT_VERSION

        safeguards = [
            safeguard
            if isinstance(safeguard, Safeguard)
            else SafeguardKind[safeguard["kind"]].value(
                **{p: v for p, v in safeguard.items() if p != "kind"}
            )
            for safeguard in safeguards
        ]

        self._pointwise_safeguards = tuple(
            safeguard
            for safeguard in safeguards
            if isinstance(safeguard, PointwiseSafeguard)
        )
        self._stencil_safeguards = tuple(
            safeguard
            for safeguard in safeguards
            if isinstance(safeguard, StencilSafeguard)
        )
        unsupported_safeguards = [
            safeguard
            for safeguard in safeguards
            if not isinstance(safeguard, (PointwiseSafeguard, StencilSafeguard))
        ]

        assert len(unsupported_safeguards) == 0, (
            f"unsupported safeguards {unsupported_safeguards!r}"
        )

    @property
    def safeguards(self) -> Collection[Safeguard]:
        """
        The collection of safeguards.
        """

        return self._pointwise_safeguards + self._stencil_safeguards

    @property
    def late_bound(self) -> Set[Parameter]:
        """
        The set of late-bound parameters that the safeguards have.

        Late-bound parameters are only bound when computing the correction, in
        contrast to the normal early-bound parameters that are configured
        during safeguard initialisation.

        Late-bound parameters can be used for parameters that depend on the
        specific data that is to be safeguarded.
        """

        return frozenset(b for s in self.safeguards for b in s.late_bound)

    @property
    def builtin_late_bound(self) -> Set[Parameter]:
        """
        The set of built-in late-bound constants that the safeguards provide
        automatically, which include `$x` and `$X`.
        """

        return frozenset([Parameter("$x"), Parameter("$X")])

    @property
    def version(self) -> str:
        """
        The version of the format of the correction computed by the
        [`compute_correction`][compression_safeguards.api.Safeguards.compute_correction]
        method.

        The safeguards can only
        [`apply_correction`][compression_safeguards.api.Safeguards.apply_correction]s
        with the matching version.
        """

        return _FORMAT_VERSION

    @staticmethod
    def supported_dtypes() -> frozenset[np.dtype]:
        """
        The set of numpy [`dtype`][numpy.dtype]s that the safeguards support.
        """

        return _SUPPORTED_DTYPES

    def correction_dtype_for_data(self, dtype: np.dtype[T]) -> np.dtype[C]:
        """
        Compute the dtype of the correction for data of the provided `dtype`.

        Parameters
        ----------
        dtype : np.dtype[T]
            The dtype of the data.

        Returns
        -------
        correction : np.dtype[C]
            The dtype of the correction.
        """

        return as_bits(np.array((), dtype=dtype)).dtype

    def compute_correction(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Mapping[str | Parameter, Value] | Bindings = Bindings.empty(),
    ) -> np.ndarray[S, np.dtype[C]]:
        """
        Compute the correction required to make the `prediction` array satisfy the safeguards relative to the `data` array.

        The `data` array must contain the complete data, i.e. not just a chunk
        of data, so that non-pointwise safeguards are correctly applied.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            The data array, relative to which the safeguards are enforced.
        prediction : np.ndarray[S, np.dtype[T]]
            The prediction array for which the correction is computed.
        late_bound : Mapping[str, Value] | Bindings
            The bindings for all late-bound parameters of the safeguards.

            The bindings must resolve all late-bound parameters and include no
            extraneous parameters.

            The safeguards automatically provide the `$x` and `$X` built-in
            constants, which must not be included.

        Returns
        -------
        correction : np.ndarray[S, np.dtype[C]]
            The correction array.
        """

        assert data.dtype in _SUPPORTED_DTYPES, (
            f"can only safeguard arrays of dtype {', '.join(d.name for d in _SUPPORTED_DTYPES)}"
        )

        if len(self._stencil_safeguards) > 0:
            assert not getattr(data, "chunked", False), (
                "computing the safeguards correction for an individual chunk in a chunked array is unsafe when using stencil safeguards since their safety requirements cannot be guaranteed across chunk boundaries"
            )

        assert data.dtype == prediction.dtype
        assert data.shape == prediction.shape

        late_bound = (
            late_bound if isinstance(late_bound, Bindings) else Bindings(**late_bound)
        )

        late_bound_reqs = self.late_bound
        late_bound_builtin = {
            p: data for p in late_bound_reqs if p in self.builtin_late_bound
        }
        late_bound_reqs = late_bound_reqs - late_bound_builtin.keys()
        late_bound_keys = frozenset(late_bound.parameters())
        assert late_bound_reqs == late_bound_keys, (
            f"late_bound is missing bindings for {sorted(late_bound_reqs - late_bound_keys)} / has extraneous bindings {sorted(late_bound_keys - late_bound_reqs)}"
        )

        if len(late_bound_builtin) > 0:
            late_bound = late_bound.update(**late_bound_builtin)  # type: ignore

        all_ok = True
        for safeguard in self.safeguards:
            if not safeguard.check(data, prediction, late_bound=late_bound):
                all_ok = False
                break

        if all_ok:
            return np.zeros_like(as_bits(data))

        all_intervals = []
        for safeguard in self._pointwise_safeguards + self._stencil_safeguards:
            intervals = safeguard.compute_safe_intervals(data, late_bound=late_bound)
            assert np.all(intervals.contains(data)), (
                f"pointwise safeguard {safeguard!r}'s intervals must contain the original data"
            )
            all_intervals.append(intervals)

        combined_intervals = all_intervals[0]
        for intervals in all_intervals[1:]:
            combined_intervals = combined_intervals.intersect(intervals)
        correction = combined_intervals.pick(prediction)

        for safeguard, intervals in zip(self.safeguards, all_intervals):
            assert np.all(intervals.contains(correction)), (
                f"{safeguard!r} interval does not contain the correction {correction!r}"
            )
            assert safeguard.check(data, correction, late_bound=late_bound), (
                f"{safeguard!r} check fails after correction {correction!r} on data {data!r}"
            )

        prediction_bits = as_bits(prediction)
        correction_bits = as_bits(correction)

        return prediction_bits - correction_bits

    def apply_correction(
        self,
        prediction: np.ndarray[S, np.dtype[T]],
        correction: np.ndarray[S, np.dtype[C]],
    ) -> np.ndarray[S, np.dtype[T]]:
        """
        Apply the `correction` to the `prediction` to satisfy the safeguards for which the `correction` was computed.

        Parameters
        ----------
        prediction : np.ndarray[S, np.dtype[T]]
            The prediction array for which the correction has been computed.
        correction : np.ndarray[S, np.dtype[C]]
            The correction array.

        Returns
        -------
        corrected : np.ndarray[S, np.dtype[T]]
            The corrected array, which satisfies the safeguards.
        """

        assert correction.shape == prediction.shape

        prediction_bits = as_bits(prediction)
        correction_bits = as_bits(correction)

        corrected = prediction_bits - correction_bits

        return corrected.view(prediction.dtype)

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguards.

        Returns
        -------
        config : dict
            Configuration of the safeguards.
        """

        return dict(
            _version=self.version,
            safeguards=[safeguard.get_config() for safeguard in self.safeguards],
        )

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """
        Instantiate the safeguards from a configuration [`dict`][dict].

        Parameters
        ----------
        config : dict
            Configuration of the safeguards.

        Returns
        -------
        safeguards : Self
            Collection of safeguards.
        """

        return cls(**config)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(safeguards={list(self.safeguards)!r})"


_FORMAT_VERSION: str = "0.1.x"


_SUPPORTED_DTYPES: frozenset[np.dtype] = frozenset(
    {
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.int32),
        np.dtype(np.int64),
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.uint32),
        np.dtype(np.uint64),
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    }
)
