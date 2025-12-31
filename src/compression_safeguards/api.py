"""
Implementation of the [`Safeguards`][compression_safeguards.api.Safeguards], which compute the correction needed to satisfy a set of [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard]s.
"""

__all__ = ["Safeguards"]

import functools
from collections.abc import Collection, Mapping, Set
from typing import Final, Literal, Self, assert_never

import numpy as np
from semver import Version
from typing_extensions import override  # MSPV 3.12

from .safeguards import SafeguardKind
from .safeguards.abc import Safeguard
from .safeguards.pointwise.abc import PointwiseSafeguard
from .safeguards.stencil import BoundaryCondition, NeighbourhoodAxis
from .safeguards.stencil.abc import StencilSafeguard
from .utils._compat import _ones, _zeros
from .utils.bindings import Bindings, Parameter, Value
from .utils.cast import as_bits
from .utils.error import (
    LateBoundParameterResolutionError,
    SafeguardsSafetyBug,
    TypeSetError,
    ctx,
)
from .utils.intervals import IntervalUnion  # noqa: TC001
from .utils.typing import JSON, C, S, T


class Safeguards:
    """
    Collection of [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard]s.

    Parameters
    ----------
    safeguards : Collection[dict[str, JSON] | Safeguard]
        The safeguards that will be applied. They can either be passed as a
        safeguard configuration [`dict`][dict] or an already initialized
        [`Safeguard`][...safeguards.abc.Safeguard].

        Please refer to the [`SafeguardKind`][...safeguards.SafeguardKind] for
        an enumeration of all supported safeguards.
    _version : ...
        The version of the safeguards. Do not provide this parameter explicitly.

    Raises
    ------
    ValueError
        if the safeguards are instantiated with a configuration from an
        incompatible version of the safeguards.
    NotImplementedError
        if the `safeguards` contain an unsupported kind of safeguard
        (currently: pointwise and stencil safeguards are supported).
    ...
        if instantiating a safeguard raises an exception.
    """

    __slots__: tuple[str, ...] = ("_pointwise_safeguards", "_stencil_safeguards")
    _pointwise_safeguards: tuple[PointwiseSafeguard, ...]
    _stencil_safeguards: tuple[StencilSafeguard, ...]

    def __init__(
        self,
        *,
        safeguards: Collection[dict[str, JSON] | Safeguard],
        _version: None | str | Version = None,
    ) -> None:
        if _version is not None:
            _version = (
                _version if isinstance(_version, Version) else Version.parse(_version)
            )
            if not _version.is_compatible(self.version):
                raise (
                    ValueError(
                        f"{_version} is not semantic-versioning-compatible with "
                        + f"the safeguards version {self.version}"
                    )
                    | ctx
                )

        safeguards_: list[Safeguard] = []
        with ctx.parameter("safeguards"):
            for i, safeguard in enumerate(safeguards):
                with ctx.index(i):
                    safeguards_.append(
                        safeguard
                        if isinstance(safeguard, Safeguard)
                        else SafeguardKind.from_config(safeguard)
                    )

        self._pointwise_safeguards = tuple(
            safeguard
            for safeguard in safeguards_
            if isinstance(safeguard, PointwiseSafeguard)
        )
        self._stencil_safeguards = tuple(
            safeguard
            for safeguard in safeguards_
            if isinstance(safeguard, StencilSafeguard)
        )
        unsupported_safeguards = [
            safeguard
            for safeguard in safeguards_
            if not isinstance(safeguard, PointwiseSafeguard | StencilSafeguard)
        ]

        if len(unsupported_safeguards) > 0:
            with ctx.parameter("safeguards"):
                raise NotImplementedError(unsupported_safeguards) | ctx

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
    def version(self) -> Version:
        """
        The semantic version [^1] of the safeguards provided by this package,
        which covers

        - the guarantees provided by the safeguards (can only be weakened in a
          new breaking major release)
        - the configurations of the safeguards (can be extended backwards-
          compatibly in a new minor release)
        - the format of the safeguards corrections (can only be changed in a
          new breaking major release)

        The [`Safeguards`][..] can only load configurations for and apply
        corrections produced by safeguards with a compatible semantic version.

        Note that the version of the safeguards may be different from the
        version of the `compression-safeguards` package, which may make changes
        to the implementation or programmatic API without needing to increase
        the version of the safeguards.

        [^1]: <https://semver.org>
        """

        return _SAFEGUARDS_VERSION

    @staticmethod
    def supported_dtypes() -> frozenset[np.dtype[np.number]]:
        """
        The set of numpy [`dtype`][numpy.dtype]s that the safeguards support.
        """

        return _SUPPORTED_DTYPES

    def correction_dtype_for_data(self, dtype: np.dtype[T]) -> np.dtype[C]:
        """
        Compute the dtype of the correction for data of the provided `dtype`.

        The correction dtype is the corresponding binary unsigned integer data
        type with the same bit size.

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

    def check(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Mapping[str | Parameter, Value] | Bindings = Bindings.EMPTY,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> bool:
        """
        Check if the `prediction` array upholds the properties enforced by the safeguards with respect to the `data` array.

        The `data` array must contain the complete data, i.e. not just a chunk
        of data, so that non-pointwise safeguards are correctly applied. Please
        use the [`check_chunk`][..check_chunk] method instead when working with
        individual chunks of data.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            The data array, relative to which the safeguards are checked.
        prediction : np.ndarray[S, np.dtype[T]]
            The prediction array for which the safeguards are checked.
        late_bound : Mapping[str | Parameter, Value] | Bindings
            The bindings for all late-bound parameters of the safeguards.

            The bindings must resolve all late-bound parameters and include no
            extraneous parameters.

            The safeguards automatically provide the `$x` and `$X` built-in
            constants, which must not be included.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only check at data points where the condition is [`True`][True].

        Returns
        -------
        ok : bool
            `True` if the check succeeded.

        Raises
        ------
        TypeSetError
            if the `data` uses an unsupported data type.
        ValueError
            if the `data`'s dtype or shape do not match the `prediction`'s.
        RuntimeError
            if the `data` array is chunked.
        LateBoundParameterResolutionError
            if `late_bound` does not resolve all late-bound parameters of the
            safeguards or includes any extraneous parameters.
        ...
            if checking a safeguard raises an exception.
        """

        late_bound = self._prepare_non_chunked_bindings(
            data=data,
            prediction=prediction,
            late_bound=late_bound,
            description="checking the safeguards",
            chunked_method_name="check_chunk",
        )

        for safeguard in self.safeguards:
            if not safeguard.check(
                data, prediction, late_bound=late_bound, where=where
            ):
                return False

        return True

    def compute_correction(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Mapping[str | Parameter, Value] | Bindings = Bindings.EMPTY,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[C]]:
        """
        Compute the correction required to make the `prediction` array satisfy the safeguards relative to the `data` array.

        The `data` array must contain the complete data, i.e. not just a chunk
        of data, so that non-pointwise safeguards are correctly applied. Please
        use the [`compute_chunked_correction`][..compute_chunked_correction]
        method instead when working with individual chunks of data.

        The correction is defined as `as_bits(corrected) - as_bits(prediction)`
        using wrapping unsigned integer arithmetic. It has the corresponding
        binary unsigned integer data type with the same bit size. The
        correction can be applied using [`apply_correction`][..apply_correction]
        to get the corrected `prediction`.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            The data array, relative to which the safeguards are enforced.
        prediction : np.ndarray[S, np.dtype[T]]
            The prediction array for which the correction is computed.
        late_bound : Mapping[str | Parameter, Value] | Bindings
            The bindings for all late-bound parameters of the safeguards.

            The bindings must resolve all late-bound parameters and include no
            extraneous parameters.

            The safeguards automatically provide the `$x` and `$X` built-in
            constants, which must not be included.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the correction at data points where the condition
            is [`True`][True].

        Returns
        -------
        correction : np.ndarray[S, np.dtype[C]]
            The correction array.

        Raises
        ------
        TypeSetError
            if the `data` uses an unsupported data type.
        ValueError
            if the `data`'s dtype or shape do not match the `prediction`'s.
        RuntimeError
            if the `data` array is chunked.
        LateBoundParameterResolutionError
            if `late_bound` does not resolve all late-bound parameters of the
            safeguards or includes any extraneous parameters.
        ...
            if computing the correction for a safeguard raises an exception.
        """
        # explicitly do not document that SafeguardsSafetyBug can be raised

        late_bound = self._prepare_non_chunked_bindings(
            data=data,
            prediction=prediction,
            late_bound=late_bound,
            description="computing the safeguards correction",
            chunked_method_name="compute_chunked_correction",
        )

        all_ok = True
        for safeguard in self.safeguards:
            if not safeguard.check(
                data, prediction, late_bound=late_bound, where=where
            ):
                all_ok = False
                break

        if all_ok:
            return _zeros(data.shape, self.correction_dtype_for_data(data.dtype))

        # ensure we don't accidentally forget to handle new kinds of safeguards here
        assert len(self.safeguards) == len(self._pointwise_safeguards) + len(
            self._stencil_safeguards
        )

        all_intervals: list[IntervalUnion[T, int, int]] = []
        for safeguard in self._pointwise_safeguards + self._stencil_safeguards:
            intervals = safeguard.compute_safe_intervals(
                data, late_bound=late_bound, where=where
            )
            if not np.all(intervals.contains(data)):
                raise (
                    SafeguardsSafetyBug(
                        f"the safe intervals for the {safeguard!r} safeguard "
                        + "do not contain the original data"
                    )
                    | ctx
                )
            all_intervals.append(intervals)

        combined_intervals = all_intervals[0]
        for intervals in all_intervals[1:]:
            combined_intervals = combined_intervals.intersect(intervals)
        corrected = combined_intervals.pick(prediction)

        for safeguard, intervals in zip(self.safeguards, all_intervals):
            if not np.all(intervals.contains(corrected)):
                raise (
                    SafeguardsSafetyBug(
                        f"the safe intervals for the {safeguard!r} safeguard "
                        + "do not contain the corrected array"
                    )
                    | ctx
                )
            if not safeguard.check(data, corrected, late_bound=late_bound, where=where):
                raise (
                    SafeguardsSafetyBug(
                        f"the check for the {safeguard!r} safeguard fails "
                        + "with the corrected array"
                    )
                    | ctx
                )

        prediction_bits: np.ndarray[S, np.dtype[C]] = as_bits(prediction)
        corrected_bits: np.ndarray[S, np.dtype[C]] = as_bits(corrected)

        with np.errstate(
            over="ignore",
            under="ignore",
        ):
            return corrected_bits - prediction_bits

    def _prepare_non_chunked_bindings(
        self,
        *,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        late_bound: Mapping[str | Parameter, Value] | Bindings,
        description: str,
        chunked_method_name: str,
    ) -> Bindings:
        TypeSetError.check_dtype_or_raise(data.dtype, _SUPPORTED_DTYPES)

        # ensure we don't accidentally forget to handle new kinds of safeguards here
        assert len(self.safeguards) == len(self._pointwise_safeguards) + len(
            self._stencil_safeguards
        )

        with ctx.parameter("prediction"):
            if prediction.dtype != data.dtype:
                raise ValueError("prediction.dtype must match data.dtype") | ctx
            if prediction.shape != data.shape:
                raise ValueError("prediction.shape must match data.shape") | ctx

        if len(self._stencil_safeguards) > 0 and getattr(data, "chunked", False):
            with ctx.parameter("data"):
                raise (
                    RuntimeError(
                        f"{description} for an individual chunk in a chunked "
                        + "array is unsafe when using stencil safeguards since "
                        + "their safety requirements cannot be guaranteed "
                        + f"across chunk boundaries; use {chunked_method_name} "
                        + "instead"
                    )
                    | ctx
                )

        late_bound = (
            late_bound if isinstance(late_bound, Bindings) else Bindings(**late_bound)
        )

        late_bound_reqs = self.late_bound
        late_bound_builtin = {
            p: data for p in late_bound_reqs if p in self.builtin_late_bound
        }
        late_bound_reqs = frozenset(late_bound_reqs - late_bound_builtin.keys())
        late_bound_keys = frozenset(late_bound.parameters())
        LateBoundParameterResolutionError.check_or_raise(
            late_bound_reqs, late_bound_keys
        )

        if len(late_bound_builtin) > 0:
            late_bound = late_bound.update(
                **{str(p): v for p, v in late_bound_builtin.items()}
            )

        return late_bound

    def apply_correction(
        self,
        prediction: np.ndarray[S, np.dtype[T]],
        correction: np.ndarray[S, np.dtype[C]],
    ) -> np.ndarray[S, np.dtype[T]]:
        """
        Apply the `correction` to the `prediction` to satisfy the safeguards for which the `correction` was computed.

        The `prediction` must be bitwise equivalent to the `prediction` that
        was used to compute the `correction`.

        This method is guaranteed to work for chunked data as well, i.e.
        applying a chunk of the `correction` to the corresponding chunk of the
        `prediction` produces the correct result.

        The `correction` is applied with
        `from_bits(as_bits(prediction) + correction)` using wrapping unsigned
        integer arithmetic.

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

        Raises
        ------
        ValueError
            if the `correction`'s shape dos not match the `prediction`'s, or if
            the `correction`'s dtype does not match the correction dtype for
            the `prediction`'s dtype.
        """

        with ctx.parameter("correction"):
            if correction.dtype != self.correction_dtype_for_data(prediction.dtype):
                raise (
                    ValueError(
                        "correction.dtype must match the correction dtype for "
                        + "prediction.dtype"
                    )
                    | ctx
                )
            if correction.shape != prediction.shape:
                raise ValueError("correction.shape must match prediction.shape") | ctx

        prediction_bits: np.ndarray[S, np.dtype[C]] = as_bits(prediction)
        correction_bits = correction

        with np.errstate(
            over="ignore",
            under="ignore",
        ):
            corrected_bits: np.ndarray[S, np.dtype[C]] = np.add(
                prediction_bits, correction_bits
            )

        return corrected_bits.view(prediction.dtype)

    @functools.lru_cache
    def compute_required_stencil_for_chunked_correction(
        self, data_shape: tuple[int, ...]
    ) -> tuple[
        tuple[
            Literal[BoundaryCondition.valid, BoundaryCondition.wrap], NeighbourhoodAxis
        ],
        ...,
    ]:
        """
        Compute the shape of the stencil neighbourhood around chunks of the complete data that is required to compute the chunked corrections.

        For each data dimension, the stencil might require either a
        [valid][....safeguards.stencil.BoundaryCondition.valid] or
        [wrapping][....safeguards.stencil.BoundaryCondition.wrap] boundary
        condition.

        This method also checks that the data shape is compatible with the
        safeguards.

        Parameters
        ----------
        data_shape : tuple[int, ...]
            The shape of the complete data.

        Returns
        -------
        stencil_shape : tuple[tuple[Literal[BoundaryCondition.valid, BoundaryCondition.wrap], NeighbourhoodAxis], ...]
            The shape of the required stencil neighbourhood around each chunk.

        Raises
        ------
        ...
            if computing the stencil neighbourhood for a safeguard raises an
            exception.
        """

        neighbourhood: list[
            tuple[
                Literal[BoundaryCondition.valid, BoundaryCondition.wrap],
                NeighbourhoodAxis,
            ]
        ] = [(BoundaryCondition.valid, NeighbourhoodAxis(0, 0)) for _ in data_shape]

        # ensure we don't accidentally forget to handle new kinds of safeguards here
        assert len(self.safeguards) == len(self._pointwise_safeguards) + len(
            self._stencil_safeguards
        )

        # pointwise safeguards don't impose any stencil neighbourhood
        #  requirements

        for safeguard in self._stencil_safeguards:
            for i, bs in enumerate(
                safeguard.compute_check_neighbourhood_for_data_shape(data_shape)
            ):
                for b, s in bs.items():
                    n_before, n_after = (
                        neighbourhood[i][1].before,
                        neighbourhood[i][1].after,
                    )

                    # we now know that the safeguards have a stencil of
                    #  [before, ..., x, ..., after]
                    # this stencil is sufficient to compute the safeguards for x
                    #
                    # BUT the elements in before and after can also back-
                    #  contribute to the safe intervals of x,
                    # so we need to ensure that all elements in the stencil can
                    #  also apply the safeguards, i.e. they also need their
                    #  stencil supplied
                    # in particular, we need to ensure that the elements before
                    #  x, which will look after elements to the right, get
                    #  their stencil, and that the elements after x, which will
                    #  look before elements to the left, get their stencil too,
                    #  since they will back-contribute their intervals to x
                    #
                    # therefore we actually need to ~double~ the stencil to
                    # [after+before, ..., before, ..., x, ..., after, ..., before+after]

                    match b:
                        case (
                            BoundaryCondition.valid
                            | BoundaryCondition.constant
                            | BoundaryCondition.edge
                        ):
                            # nothing special, but we do need to extend the stencil
                            neighbourhood[i] = (
                                neighbourhood[i][0],
                                NeighbourhoodAxis(
                                    before=max(n_before, s.after + s.before),
                                    after=max(n_after, s.before + s.after),
                                ),
                            )
                        case BoundaryCondition.reflect:
                            # reflect:           [ 1, 2, 3 ]
                            #       -> [..., 3, 2, 1, 2, 3, 2, 1, ...]
                            # worst case, the reflection on the left exits the
                            #  chunk on the right, and same for on the right
                            # so we need to extend with max(before, after) at
                            #  both ends
                            # if we have before=1 and after=0 and focus on x=3,
                            #  this would give us the stencil 2, 3|
                            # however, when the reflect boundary is applied, it
                            #  would be 3, 2, 3| and 2 would now look at 3 and
                            #  back-contribute to 3 even though it should not
                            # if we have before=2 and after=0 and focus on x=3
                            #  in -1, 0, 1, 2, 3 then 1, 2, 3| would become
                            #  3, 2, 1, 2, 3| which is incorrect; 0, 1, 2, 3|
                            #  would become 2, 1, 0, 1, 2, 3 which is also
                            #  incorrect but good enough to no longer back-
                            #  contribute
                            # therefore, we need to extend by at least one
                            #  beyond the stencil if the stencil is non-zero
                            # but since we need before+after on both sides, the
                            #  max cancels out (and we also cannot optimise
                            #  based on the size of the chunk)
                            neighbourhood[i] = (
                                neighbourhood[i][0],
                                NeighbourhoodAxis(
                                    before=max(
                                        n_before,
                                        max(s.after, bool(s.before)) + s.before,
                                    ),
                                    after=max(
                                        n_after, max(s.before, bool(s.after)) + s.after
                                    ),
                                ),
                            )
                        case BoundaryCondition.symmetric:
                            # symmetric:         [ 1, 2, 3 ]
                            #       -> [..., 2, 1, 1, 2, 3, 3, 2, ...]
                            # similar to reflect, but the edge is repeated,
                            #  meaning that accidental back-contribution is not
                            #  possible
                            # worst case, the symmetry on the left exits the
                            #  chunk on the right, and same for on the right
                            # but since we need before+after on both sides, the
                            #  max cancels out (and we also cannot optimise
                            #  based on the size of the chunk)
                            neighbourhood[i] = (
                                neighbourhood[i][0],
                                NeighbourhoodAxis(
                                    before=max(n_before, s.after + s.before),
                                    after=max(n_after, s.before + s.after),
                                ),
                            )
                        case BoundaryCondition.wrap:
                            # we need to extend the stencil and tell xarray and
                            #  remember that we will need a wrapping / periodic
                            #  boundary
                            neighbourhood[i] = (
                                # only require the wrapping boundary condition
                                #  if there actually is a boundary
                                BoundaryCondition.wrap
                                if s.before > 0 or s.after > 0
                                else neighbourhood[i][0],
                                NeighbourhoodAxis(
                                    before=max(n_before, s.after + s.before),
                                    after=max(n_after, s.before + s.after),
                                ),
                            )
                        case _:
                            assert_never(b)

        return tuple(neighbourhood)

    def check_chunk(
        self,
        data_chunk: np.ndarray[S, np.dtype[T]],
        prediction_chunk: np.ndarray[S, np.dtype[T]],
        *,
        data_shape: tuple[int, ...],
        chunk_offset: tuple[int, ...],
        chunk_stencil: tuple[
            tuple[
                Literal[BoundaryCondition.valid, BoundaryCondition.wrap],
                NeighbourhoodAxis,
            ],
            ...,
        ],
        late_bound_chunk: Mapping[str | Parameter, Value] | Bindings = Bindings.EMPTY,
        where_chunk: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> bool:
        """
        Check if the `prediction_chunk` array chunk upholds the properties enforced by the safeguards with respect to the `data_chunk` array chunk.

        Both the `data_chunk` and `prediction_chunk` contain the stencil around
        the chunk, and the shape of this applied stencil is specified in the
        `chunk_stencil` parameter. This stencil must be compatible with the
        required stencil returned by
        [`compute_required_stencil_for_chunked_correction(data_shape)`][..compute_required_stencil_for_chunked_correction]:
        - a wrapping boundary is always compatible with a valid boundary.
        - a larger stencil is always compatible with a smaller stencil.
        - a smaller stencil is sometimes compatible with a larger stencil, iff
          the smaller stencil is near the entire data boundary and still
          includes all required elements; for instance, providing the entire
          data as a single chunk with no stencil is always compatible with any
          stencil

        This advanced method should only be used when working with individual
        chunks of data, for non-chunked data please use the simpler and more
        efficient [`check`][..check] method instead.

        Parameters
        ----------
        data_chunk : np.ndarray[S, np.dtype[T]]
            A stencil-extended chunk from the data array, relative to which the
            safeguards are checked.
        prediction_chunk : np.ndarray[S, np.dtype[T]]
            The corresponding stencil-extended chunk from the prediction array
            for which the safeguards are checked.
        data_shape : tuple[int, ...]
            The shape of the entire data array, i.e. not just the chunk,
            without any stencil.
        chunk_offset : tuple[int, ...]
            The offset of the non-stencil-extended chunk inside the entire
            array. For arrays going from left to right, bottom to top, ..., the
            offset is the index of the bottom left element in the entire array.
        chunk_stencil : tuple[tuple[Literal[BoundaryCondition.valid, BoundaryCondition.wrap], NeighbourhoodAxis], ...]
            The shape of the stencil neighbourhood that was applied around the
            chunk in `data_chunk` and `prediction_chunk`.
        late_bound_chunk : Mapping[str | Parameter, Value] | Bindings
            The bindings for all late-bound parameters of the safeguards.

            The bindings must resolve all late-bound parameters and include no
            extraneous parameters.

            If a binding resolves to an array, it must be the corresponding
            chunk of the entire late-bound array.

            The safeguards automatically provide the `$x` and `$X` built-in
            constants, which must not be included.
        where_chunk : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only check at data points where the condition is [`True`][True].

        Returns
        -------
        chunk_ok : bool
            `True` if the check succeeded for the chunk.

        Raises
        ------
        TypeSetError
            if the `data_chunk` uses an unsupported data type.
        ValueError
            if the `data_chunk`'s dtype or shape do not match the
            `prediction_chunk`'s, or if the length of the `data_shape`,
            `chunk_offset`, or `chunk_stencil` do not match the dimensionality
            of the `data`.
        ValueError
            if the `chunk_stencil` is not compatible with the required stencil.
        LateBoundParameterResolutionError
            if `late_bound_chunk` does not resolve all late-bound parameters of
            the safeguards or includes any extraneous parameters.
        ValueError
            if any `late_bound_chunk` array could not be broadcast to the
            `data_chunk`'s shape.
        ...
            if checking a safeguard raises an exception.
        """

        (
            data_chunk_,
            prediction_chunk_,
            late_bound_chunk,
            where_chunk_,
            non_stencil_indices,
        ) = self._prepare_stencil_chunked_arrays_and_bindings(
            data_chunk=data_chunk,
            prediction_chunk=prediction_chunk,
            data_shape=data_shape,
            chunk_offset=chunk_offset,
            chunk_stencil=chunk_stencil,
            late_bound_chunk=late_bound_chunk,
            where_chunk=where_chunk,
        )

        # ensure we don't accidentally forget to handle new kinds of safeguards here
        assert len(self.safeguards) == len(self._pointwise_safeguards) + len(
            self._stencil_safeguards
        )

        all_ok = _ones(
            data_chunk_[tuple(non_stencil_indices)].shape, dtype=np.dtype(np.bool)
        )

        # we need to use pointwise checks here so that we can only look at the
        #  non-stencil check results
        for safeguard in self._pointwise_safeguards + self._stencil_safeguards:
            all_ok &= safeguard.check_pointwise(
                data_chunk_,
                prediction_chunk_,
                late_bound=late_bound_chunk,
                where=where_chunk_,
            )[tuple(non_stencil_indices)]

            if not np.all(all_ok):
                return False

        return True

    def compute_chunked_correction(
        self,
        data_chunk: np.ndarray[S, np.dtype[T]],
        prediction_chunk: np.ndarray[S, np.dtype[T]],
        *,
        data_shape: tuple[int, ...],
        chunk_offset: tuple[int, ...],
        chunk_stencil: tuple[
            tuple[
                Literal[BoundaryCondition.valid, BoundaryCondition.wrap],
                NeighbourhoodAxis,
            ],
            ...,
        ],
        any_chunk_check_failed: bool,
        late_bound_chunk: Mapping[str | Parameter, Value] | Bindings = Bindings.EMPTY,
        where_chunk: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[tuple[int, ...], np.dtype[C]]:
        """
        Compute the correction required to make the `prediction_chunk` array chunk satisfy the safeguards relative to the `data_chunk` array chunk.

        Both the `data_chunk` and `prediction_chunk` contain the stencil around
        the chunk, and the shape of this applied stencil is specified in the
        `chunk_stencil` parameter. This stencil must be compatible with the
        required stencil returned by
        [`compute_required_stencil_for_chunked_correction(data_shape)`][..compute_required_stencil_for_chunked_correction]:
        - a wrapping boundary is always compatible with a valid boundary.
        - a larger stencil is always compatible with a smaller stencil.
        - a smaller stencil is sometimes compatible with a larger stencil, iff
          the smaller stencil is near the entire data boundary and still
          includes all required elements; for instance, providing the entire
          data as a single chunk with no stencil is always compatible with any
          stencil

        This advanced method should only be used when working with individual
        chunks of data, for non-chunked data please use the simpler and more
        efficient [`compute_correction`][..compute_correction] method instead.

        The (chunked) correction is defined as
        `as_bits(corrected) - as_bits(prediction)` using wrapping unsigned
        integer arithmetic. It has the corresponding binary unsigned integer
        data type with the same bit size. The chunked correction can be applied
        using [`apply_correction`][..apply_correction] to get the corrected
        chunked `prediction`.

        Parameters
        ----------
        data_chunk : np.ndarray[S, np.dtype[T]]
            A stencil-extended chunk from the data array, relative to which the
            safeguards are enforced.
        prediction_chunk : np.ndarray[S, np.dtype[T]]
            The corresponding stencil-extended chunk from the prediction array
            for which the correction is computed.
        data_shape : tuple[int, ...]
            The shape of the entire data array, i.e. not just the chunk.
        chunk_offset : tuple[int, ...]
            The offset of the non-stencil-extended chunk inside the entire
            array. For arrays going from left to right, bottom to top, ..., the
            offset is the index of the bottom left element in the entire array.
        chunk_stencil : tuple[tuple[Literal[BoundaryCondition.valid, BoundaryCondition.wrap], NeighbourhoodAxis], ...]
            The shape of the stencil neighbourhood that was applied around the
            chunk in `data_chunk` and `prediction_chunk`.
        late_bound_chunk : Mapping[str | Parameter, Value] | Bindings
            The bindings for all late-bound parameters of the safeguards.

            The bindings must resolve all late-bound parameters and include no
            extraneous parameters.

            If a binding resolves to an array, it must be the corresponding
            chunk of the entire late-bound array.

            The safeguards automatically provide the `$x` and `$X` built-in
            constants, which must not be included.
        where_chunk : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the correction at data points where the condition is
            [`True`][True].

        Returns
        -------
        correction_chunk : np.ndarray[tuple[int, ...], np.dtype[C]]
            The correction array chunk. The correction chunk is truncated to
            remove the stencil, i.e. it only contains the correction for the
            non-stencil-extended chunk.

        Raises
        ------
        TypeSetError
            if the `data_chunk` uses an unsupported data type.
        ValueError
            if the `data_chunk`'s dtype or shape do not match the
            `prediction_chunk`'s, or if the length of the `data_shape`,
            `chunk_offset`, or `chunk_stencil` do not match the dimensionality
            of the `data`.
        ValueError
            if the `chunk_stencil` is not compatible with the required stencil.
        LateBoundParameterResolutionError
            if `late_bound_chunk` does not resolve all late-bound parameters of
            the safeguards or includes any extraneous parameters.
        ValueError
            if any `late_bound_chunk` array could not be broadcast to the
            `data_chunk`'s shape.
        ...
            if computing the correction for a safeguard raises an exception.
        """
        # explicitly do not document that SafeguardsSafetyBug can be raised

        (
            data_chunk_,
            prediction_chunk_,
            late_bound_chunk,
            where_chunk_,
            non_stencil_indices,
        ) = self._prepare_stencil_chunked_arrays_and_bindings(
            data_chunk=data_chunk,
            prediction_chunk=prediction_chunk,
            data_shape=data_shape,
            chunk_offset=chunk_offset,
            chunk_stencil=chunk_stencil,
            late_bound_chunk=late_bound_chunk,
            where_chunk=where_chunk,
        )

        # if no chunk requires a correction, this one doesn't either
        if not any_chunk_check_failed:
            return _zeros(
                data_chunk_[tuple(non_stencil_indices)].shape,
                self.correction_dtype_for_data(data_chunk_.dtype),
            )

        safeguard: Safeguard

        # if only pointwise safeguards are used, check if we need to correct
        #  this chunk
        if len(self._pointwise_safeguards) == len(self.safeguards):
            all_ok = _ones(
                data_chunk_[tuple(non_stencil_indices)].shape, dtype=np.dtype(np.bool)
            )

            # we need to use pointwise checks here so that we can only look at the
            #  non-stencil check results
            for safeguard in self._pointwise_safeguards:
                all_ok &= safeguard.check_pointwise(
                    data_chunk_,
                    prediction_chunk_,
                    late_bound=late_bound_chunk,
                    where=where_chunk_,
                )[tuple(non_stencil_indices)]

                if not np.all(all_ok):
                    break

            if np.all(all_ok):
                return _zeros(
                    data_chunk_[tuple(non_stencil_indices)].shape,
                    self.correction_dtype_for_data(data_chunk_.dtype),
                )

        # otherwise, correct the chunk
        # if stencil safeguards are used, then any chunk needing a correction
        #  requires all chunks to be corrected

        # ensure we don't accidentally forget to handle new kinds of safeguards here
        assert len(self.safeguards) == len(self._pointwise_safeguards) + len(
            self._stencil_safeguards
        )

        all_intervals: list[IntervalUnion[T, int, int]] = []
        for safeguard in self._pointwise_safeguards + self._stencil_safeguards:
            intervals = safeguard.compute_safe_intervals(
                data_chunk_, late_bound=late_bound_chunk, where=where_chunk_
            )
            if not np.all(intervals.contains(data_chunk_)):
                raise (
                    SafeguardsSafetyBug(
                        f"the safe intervals for the {safeguard!r} safeguard "
                        + "must contain the original data chunk"
                    )
                    | ctx
                )
            all_intervals.append(intervals)

        combined_intervals = all_intervals[0]
        for intervals in all_intervals[1:]:
            combined_intervals = combined_intervals.intersect(intervals)
        corrected_chunk = combined_intervals.pick(prediction_chunk_)

        for safeguard, intervals in zip(self.safeguards, all_intervals):
            if not np.all(intervals.contains(corrected_chunk)):
                raise (
                    SafeguardsSafetyBug(
                        f"the safe intervals for the {safeguard!r} safeguard "
                        + "do not contain the corrected array chunk"
                    )
                    | ctx
                )
            if not safeguard.check(
                data_chunk_,
                corrected_chunk,
                late_bound=late_bound_chunk,
                where=where_chunk_,
            ):
                raise (
                    SafeguardsSafetyBug(
                        f"the check for the {safeguard!r} safeguard fails "
                        + "with the corrected array chunk"
                    )
                    | ctx
                )

        prediction_chunk_bits: np.ndarray[tuple[int, ...], np.dtype[C]] = as_bits(
            prediction_chunk_
        )
        corrected_chunk_bits: np.ndarray[tuple[int, ...], np.dtype[C]] = as_bits(
            corrected_chunk
        )

        with np.errstate(
            over="ignore",
            under="ignore",
        ):
            return (corrected_chunk_bits - prediction_chunk_bits)[
                tuple(non_stencil_indices)
            ]

    def _prepare_stencil_chunked_arrays_and_bindings(
        self,
        *,
        data_chunk: np.ndarray[S, np.dtype[T]],
        prediction_chunk: np.ndarray[S, np.dtype[T]],
        data_shape: tuple[int, ...],
        chunk_offset: tuple[int, ...],
        chunk_stencil: tuple[
            tuple[
                Literal[BoundaryCondition.valid, BoundaryCondition.wrap],
                NeighbourhoodAxis,
            ],
            ...,
        ],
        late_bound_chunk: Mapping[str | Parameter, Value] | Bindings,
        where_chunk: Literal[True] | np.ndarray[S, np.dtype[np.bool]],
    ) -> tuple[
        np.ndarray[tuple[int, ...], np.dtype[T]],
        np.ndarray[tuple[int, ...], np.dtype[T]],
        Bindings,
        Literal[True] | np.ndarray[tuple[int, ...], np.dtype[np.bool]],
        tuple[slice, ...],
    ]:
        TypeSetError.check_dtype_or_raise(data_chunk.dtype, _SUPPORTED_DTYPES)

        with ctx.parameter("prediction_chunk"):
            if prediction_chunk.dtype != data_chunk.dtype:
                raise (
                    ValueError("prediction_chunk.dtype must match data_chunk.dtype")
                    | ctx
                )
            if prediction_chunk.shape != data_chunk.shape:
                raise (
                    ValueError("prediction_chunk.shape must match data_chunk.shape")
                    | ctx
                )
        with ctx.parameter("data_shape"):
            if len(data_shape) != data_chunk.ndim:
                raise ValueError("len(data_shape) must match data_chunk.ndim") | ctx
        with ctx.parameter("chunk_offset"):
            if len(chunk_offset) != data_chunk.ndim:
                raise ValueError("len(chunk_offset) must match data_chunk.ndim") | ctx
        with ctx.parameter("chunk_stencil"):
            if len(chunk_stencil) != data_chunk.ndim:
                raise ValueError("len(chunk_stencil) must match data_chunk.ndim") | ctx

        chunk_shape: tuple[int, ...] = tuple(
            a - s[1].before - s[1].after
            for a, s in zip(data_chunk.shape, chunk_stencil)
        )

        required_stencil = self.compute_required_stencil_for_chunked_correction(
            data_shape
        )

        stencil_indices: list[slice] = []
        stencil_roll: list[int] = []
        non_stencil_indices: list[slice] = []

        # (1): check that the chunk stencil is compatible with the required
        #       stencil
        #      this is not trivial since we need to account for huge chunks
        #       where downgrading the stencil can work out
        # (2): compute indices to extract just the needed data and data+stencil
        with ctx.parameter("chunk_stencil"):
            for i, (c, r) in enumerate(zip(chunk_stencil, required_stencil)):
                with ctx.index(i):
                    # complete chunks that span the entire data along the axis
                    #  are always allowed
                    if (
                        c[0] in (BoundaryCondition.valid, BoundaryCondition.wrap)
                        and c[1].before == 0
                        and c[1].after == 0
                        and chunk_shape[i] == data_shape[i]
                    ):
                        stencil_indices.append(slice(None))
                        stencil_roll.append(0)
                        non_stencil_indices.append(slice(None))
                        continue

                    match c[0]:
                        case BoundaryCondition.valid:
                            # we need to check that we requested a valid
                            #  boundary, which is only compatible with itself
                            # and that the stencil is large enough
                            if r[0] != BoundaryCondition.valid:
                                raise (
                                    ValueError(
                                        f"requires {r[0].name} stencil "
                                        + "boundary but found valid boundary"
                                    )
                                    | ctx
                                )

                            # what is the required stencil after adjusting for
                            #  near-boundary stencil truncation?
                            rs = NeighbourhoodAxis(
                                before=min(chunk_offset[i], r[1].before),
                                after=min(
                                    r[1].after,
                                    data_shape[i] - chunk_shape[i] - chunk_offset[i],
                                ),
                            )
                            if c[1].before < rs.before:
                                with ctx.parameter("before"):
                                    raise (
                                        ValueError(
                                            "requires stencil with at least "
                                            + f"{rs.before} point(s) before "
                                            + f"but received only {c[1].before}"
                                        )
                                        | ctx
                                    )
                            if c[1].after < rs.after:
                                with ctx.parameter("after"):
                                    raise (
                                        ValueError(
                                            "requires stencil with at least "
                                            + f"{rs.after} point(s) after but "
                                            + f"received only {c[1].after}"
                                        )
                                        | ctx
                                    )

                            stencil_indices.append(
                                slice(
                                    c[1].before - rs.before,
                                    None
                                    if c[1].after == rs.after
                                    else rs.after - c[1].after,
                                )
                            )
                            stencil_roll.append(0)
                            non_stencil_indices.append(
                                slice(rs.before, None if rs.after == 0 else -rs.after)
                            )
                        case BoundaryCondition.wrap:
                            # a wrapping boundary is compatible with any other
                            #  boundary
                            if r[0] not in (
                                BoundaryCondition.valid,
                                BoundaryCondition.wrap,
                            ):
                                raise (
                                    ValueError(
                                        f"requires {r[0].name} stencil "
                                        + "boundary but found wrapping boundary"
                                    )
                                    | ctx
                                )

                            # what is the required stencil after adjusting for
                            #  near-boundary stencil truncation?
                            # (a) if the chunk is in the middle, where no
                            #     boundary condition is applied, we just keep
                            #     the stencil as-is
                            # (b) if the chunk's stencil only overlaps with the
                            #     boundary on one side, we keep the stencil on
                            #     that side as-is, as long as it does not
                            #     overlap, after wrap-around, with the stencil
                            #     on the other side in this case, we also need
                            #     to roll the stencil on the side of the data
                            #     boundary to the other side, to ensure that
                            #     other boundary conditions also see a boundary
                            # (c) otherwise, we can have the full data and
                            #     remove any excessive stencil so that the
                            #     per-safeguard stencil correctly sees how
                            #     points wrap
                            rs = NeighbourhoodAxis(
                                before=(
                                    r[1].before  # (a)
                                    if r[1].before <= chunk_offset[i]
                                    else (
                                        min(  # (b)
                                            r[1].before - chunk_offset[i],
                                            data_shape[i]
                                            - chunk_offset[i]
                                            - chunk_shape[i]
                                            - r[1].after,
                                        )
                                        if r[1].after
                                        <= (
                                            data_shape[i]
                                            - chunk_shape[i]
                                            - chunk_offset[i]
                                        )
                                        else min(chunk_offset[i], r[1].before)  # (c)
                                    )
                                ),
                                after=(
                                    r[1].after  # (a)
                                    if r[1].after
                                    <= (
                                        data_shape[i] - chunk_shape[i] - chunk_offset[i]
                                    )
                                    else (
                                        min(  # (b)
                                            chunk_offset[i]
                                            + chunk_shape[i]
                                            + r[1].after
                                            - data_shape[i],
                                            chunk_offset[i] - r[1].before,
                                        )
                                        if r[1].before <= chunk_offset[i]
                                        else min(  # (c)
                                            r[1].after,
                                            data_shape[i]
                                            - chunk_shape[i]
                                            - chunk_offset[i],
                                        )
                                    )
                                ),
                            )
                            if c[1].before < rs.before:
                                with ctx.parameter("before"):
                                    raise (
                                        ValueError(
                                            "requires stencil with at least "
                                            + f"{rs.before} point(s) before "
                                            + f"but received only {c[1].before}"
                                        )
                                        | ctx
                                    )
                            if c[1].after < rs.after:
                                with ctx.parameter("after"):
                                    raise (
                                        ValueError(
                                            "requires stencil with at least "
                                            + f"{rs.after} point(s) after but "
                                            + f"received only {c[1].after}"
                                        )
                                        | ctx
                                    )

                            roll_before = max(0, rs.before - chunk_offset[i])
                            roll_after = max(
                                0,
                                chunk_offset[i]
                                + chunk_shape[i]
                                + rs.after
                                - data_shape[i],
                            )
                            if (roll_before > 0) and (roll_after > 0):
                                roll_before, roll_after = 0, 0

                            stencil_indices.append(
                                slice(
                                    c[1].before - rs.before,
                                    None
                                    if c[1].after == rs.after
                                    else rs.after - c[1].after,
                                )
                            )
                            stencil_roll.append(-roll_before + roll_after)
                            nsi_before = rs.before + roll_after - roll_before
                            nsi_after = rs.after + roll_before - roll_after
                            non_stencil_indices.append(
                                slice(
                                    nsi_before, None if nsi_after == 0 else -nsi_after
                                )
                            )
                        case _:
                            assert_never(c[0])

        data_chunk_ = np.roll(
            data_chunk[tuple(stencil_indices)],
            shift=tuple(stencil_roll),
            axis=tuple(range(data_chunk.ndim)),
        )
        prediction_chunk_ = np.roll(
            prediction_chunk[tuple(stencil_indices)],
            shift=tuple(stencil_roll),
            axis=tuple(range(prediction_chunk.ndim)),
        )

        # create the late-bound bindings for the chunk
        late_bound_chunk = (
            late_bound_chunk
            if isinstance(late_bound_chunk, Bindings)
            else Bindings(**late_bound_chunk)
        )
        # check that all late-bound parameters have the right shape
        late_bound_chunk.expect_broadcastable_to(data_chunk.shape)
        # apply the stencil indices to the late-bound parameters
        late_bound_chunk = late_bound_chunk.apply_slice_index(tuple(stencil_indices))
        late_bound_chunk = late_bound_chunk.apply_roll(tuple(stencil_roll))

        late_bound_reqs = self.late_bound
        late_bound_builtin = {
            p: data_chunk_ for p in late_bound_reqs if p in self.builtin_late_bound
        }
        late_bound_reqs = frozenset(late_bound_reqs - late_bound_builtin.keys())
        late_bound_keys = frozenset(late_bound_chunk.parameters())
        LateBoundParameterResolutionError.check_or_raise(
            late_bound_reqs, late_bound_keys
        )

        if len(late_bound_builtin) > 0:
            late_bound_chunk = late_bound_chunk.update(
                **{str(p): v for p, v in late_bound_builtin.items()}
            )

        where_chunk_: Literal[True] | np.ndarray[tuple[int, ...], np.dtype[np.bool]] = (
            True
            if where_chunk is True
            else np.roll(
                where_chunk[tuple(stencil_indices)],
                shift=tuple(stencil_roll),
                axis=tuple(range(where_chunk.ndim)),
            )
        )

        return (
            data_chunk_,
            prediction_chunk_,
            late_bound_chunk,
            where_chunk_,
            tuple(non_stencil_indices),
        )

    def get_config(self) -> dict[str, JSON]:
        """
        Returns the configuration of the safeguards.

        Returns
        -------
        config : dict[str, JSON]
            Configuration of the safeguards.
        """

        return dict(
            safeguards=[safeguard.get_config() for safeguard in self.safeguards],
            _version=str(self.version),
        )

    @classmethod
    def from_config(cls, config: dict[str, JSON]) -> Self:
        """
        Instantiate the safeguards from a configuration [`dict`][dict].

        Parameters
        ----------
        config : dict[str, JSON]
            Configuration of the safeguards.

        Returns
        -------
        safeguards : Self
            Collection of safeguards.
        """

        return cls(**config)  # type: ignore

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(safeguards={list(self.safeguards)!r})"


_SAFEGUARDS_VERSION: Final[Version] = Version(1, 0, 0)


_SUPPORTED_DTYPES: Final[frozenset[np.dtype[np.number]]] = frozenset(
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
