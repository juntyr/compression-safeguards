"""
Implementation of the [`Safeguards`][compression_safeguards.api.Safeguards], which compute the correction needed to satisfy a set of [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard]s.
"""

__all__ = ["Safeguards"]

import functools
from collections.abc import Collection, Mapping, Set
from typing import Literal

import numpy as np
from typing_extensions import Self, assert_never  # MSPV 3.11

from .safeguards import SafeguardKind
from .safeguards.abc import Safeguard
from .safeguards.pointwise.abc import PointwiseSafeguard
from .safeguards.stencil import BoundaryCondition, NeighbourhoodAxis
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
            if not isinstance(safeguard, PointwiseSafeguard | StencilSafeguard)
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

    def check(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Mapping[str | Parameter, Value] | Bindings = Bindings.empty(),
    ) -> bool:
        """
        Check if the `prediction` array upholds the properties enforced by the safeguards with respect to the `data` array.

        The `data` array must contain the complete data, i.e. not just a chunk
        of data, so that non-pointwise safeguards are correctly applied. Please
        use the
        [`check_chunk`][compression_safeguards.api.Safeguards.check_chunk]
        method instead when working with individual chunks of data.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            The data array, relative to which the safeguards are enforced.
        prediction : np.ndarray[S, np.dtype[T]]
            The prediction array for which the safeguards are checked.
        late_bound : Mapping[str | Parameter, Value] | Bindings
            The bindings for all late-bound parameters of the safeguards.

            The bindings must resolve all late-bound parameters and include no
            extraneous parameters.

            The safeguards automatically provide the `$x` and `$X` built-in
            constants, which must not be included.

        Returns
        -------
        ok : bool
            `True` if the check succeeded.
        """

        assert data.dtype in _SUPPORTED_DTYPES, (
            f"can only safeguard arrays of dtype {', '.join(d.str for d in _SUPPORTED_DTYPES)}"
        )

        if len(self._stencil_safeguards) > 0:
            assert not getattr(data, "chunked", False), (
                "checking the safeguards for an individual chunk in a chunked "
                "array is unsafe when using stencil safeguards since their "
                "safety requirements cannot be guaranteed across chunk "
                "boundaries; use check_chunk instead"
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
            f"late_bound is missing bindings for {sorted(late_bound_reqs - late_bound_keys)} "
            f"/ has extraneous bindings {sorted(late_bound_keys - late_bound_reqs)}"
        )

        if len(late_bound_builtin) > 0:
            late_bound = late_bound.update(**late_bound_builtin)  # type: ignore

        for safeguard in self.safeguards:
            if not safeguard.check(data, prediction, late_bound=late_bound):
                return False

        return True

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
        of data, so that non-pointwise safeguards are correctly applied. Please
        use the
        [`compute_chunked_correction`][compression_safeguards.api.Safeguards.compute_chunked_correction]
        method instead when working with individual chunks of data.

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
                "computing the safeguards correction for an individual chunk "
                "in a chunked array is unsafe when using stencil safeguards "
                "since their safety requirements cannot be guaranteed across "
                "chunk boundaries; use compute_chunked_correction instead"
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
            f"late_bound is missing bindings for {sorted(late_bound_reqs - late_bound_keys)} "
            f"/ has extraneous bindings {sorted(late_bound_keys - late_bound_reqs)}"
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

        # ensure we don't accidentally forget to handle new kinds of safeguards here
        assert len(self.safeguards) == len(self._pointwise_safeguards) + len(
            self._stencil_safeguards
        )

        all_intervals = []
        for safeguard in self._pointwise_safeguards + self._stencil_safeguards:
            intervals = safeguard.compute_safe_intervals(data, late_bound=late_bound)
            assert np.all(intervals.contains(data)), (
                f"safeguard {safeguard!r}'s intervals must contain the original data"
            )
            all_intervals.append(intervals)

        combined_intervals = all_intervals[0]
        for intervals in all_intervals[1:]:
            combined_intervals = combined_intervals.intersect(intervals)
        correction = combined_intervals.pick(prediction)

        for safeguard, intervals in zip(self.safeguards, all_intervals):
            assert np.all(intervals.contains(correction)), (
                f"safeguard {safeguard!r} interval does not contain the correction {correction!r}"
            )
            assert safeguard.check(data, correction, late_bound=late_bound), (
                f"safeguard {safeguard!r} check fails after correction {correction!r} on data {data!r}"
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

        This method is guaranteed to work for chunked data as well, i.e.
        applying a chunk of the `correction` to the corresponding chunk of the
        `prediction` produces the correct result.

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

    @functools.lru_cache
    def compute_required_stencil_for_chunked_correction(
        self,
        data_shape: tuple[int, ...],
        *,
        smallest_chunk_shape: None | tuple[int, ...] = None,
    ) -> tuple[
        tuple[
            Literal[BoundaryCondition.valid, BoundaryCondition.wrap], NeighbourhoodAxis
        ],
        ...,
    ]:
        """
        Compute the shape of the stencil neighbourhood around chunks of the complete data that is required to compute the chunked corrections.

        For each data dimension, the stencil might require either a
        [valid][compression_safeguards.safeguards.stencil.BoundaryCondition.valid]
        or
        [wrapping][compression_safeguards.safeguards.stencil.BoundaryCondition.wrap]
        boundary condition.

        This method also checks that the data shape is compatible with the
        safeguards.

        Parameters
        ----------
        data_shape : tuple[int, ...]
            The shape of the complete data.
        smallest_chunk_shape : None | tuple[int, ...]
            The shape of the smallest chunk over which the chunked corrections
            will be computed, if known.

        Returns
        -------
        stencil_shape : tuple[tuple[Literal[BoundaryCondition.valid, BoundaryCondition.wrap], NeighbourhoodAxis], ...]
            The shape of the required stencil neighbourhood around each chunk.
        """

        # if the shape of the smallest chunk is not known, assume the worst
        if smallest_chunk_shape is None:
            smallest_chunk_shape = tuple(0 for _ in data_shape)

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
                    #
                    # therefore we actually need to double the stencil to
                    # [before-before, ..., before, ..., x, ..., after, ..., after+after]

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
                                    before=max(n_before, s.before * 2),
                                    after=max(n_after, s.after * 2),
                                ),
                            )
                        case BoundaryCondition.reflect:
                            # reflect:           [ 1, 2, 3 ]
                            #       -> [..., 3, 2, 1, 2, 3, 2, 1, ...]
                            # worst case, the reflection on the left exits the
                            #  chunk on the right, and same for on the right
                            # so we need to extend with max(before, after) at
                            #  both ends
                            neighbourhood[i] = (
                                neighbourhood[i][0],
                                NeighbourhoodAxis(
                                    before=max(
                                        n_before,
                                        max(
                                            s.before * 2,
                                            s.after * 2 - smallest_chunk_shape[i],
                                        ),
                                    ),
                                    after=max(
                                        n_after,
                                        max(
                                            s.before * 2 - smallest_chunk_shape[i],
                                            s.after * 2,
                                        ),
                                    ),
                                ),
                            )
                        case BoundaryCondition.symmetric:
                            # symmetric:         [ 1, 2, 3 ]
                            #       -> [..., 2, 1, 1, 2, 3, 3, 2, ...]
                            # similar to reflect, but the edge is repeated
                            neighbourhood[i] = (
                                neighbourhood[i][0],
                                NeighbourhoodAxis(
                                    before=max(
                                        n_before,
                                        max(
                                            s.before * 2,
                                            s.after * 2 - 1 - smallest_chunk_shape[i],
                                        ),
                                    ),
                                    after=max(
                                        n_after,
                                        max(
                                            s.before * 2 - 1 - smallest_chunk_shape[i],
                                            s.after * 2,
                                        ),
                                    ),
                                ),
                            )
                        case BoundaryCondition.wrap:
                            # we need to extend the stencil and tell xarray and
                            #  remember that we will need a wrapping / periodic
                            #  boundary
                            neighbourhood[i] = (
                                BoundaryCondition.wrap,
                                NeighbourhoodAxis(
                                    before=max(n_before, s.before * 2),
                                    after=max(n_after, s.after * 2),
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
        late_bound_chunk: Mapping[str | Parameter, Value] | Bindings = Bindings.empty(),
    ) -> bool:
        """
        Check if the `prediction_chunk` array chunk upholds the properties enforced by the safeguards with respect to the `data_chunk` array chunk.

        This method should only be used when working with individual chunks of
        data, otherwise please use the more efficient
        [`check`][compression_safeguards.api.Safeguards.check]
        method instead.

        Parameters
        ----------
        data_chunk : np.ndarray[S, np.dtype[T]]
            A chunk from the data array, relative to which the safeguards are
            enforced.
        prediction_chunk : np.ndarray[S, np.dtype[T]]
            The corresponding chunk from the prediction array for which the
            safeguards are checked.
        data_shape : tuple[int, ...]
            The shape of the entire data array, i.e. not just the chunk.
        chunk_offset : tuple[int, ...]
            The offset of the chunk inside the entire array. For arrays going
            from left to right, bottom to top, ... the offset is the index of
            the bottom left element in the entire array.
        chunk_stencil : tuple[tuple[Literal[BoundaryCondition.valid, BoundaryCondition.wrap], NeighbourhoodAxis], ...]
            The shape of the stencil neighbourhood around the chunk. This
            stencil must be compatible with the required stencil returned by
            [`compute_required_stencil_for_chunked_correction(data_shape)`][compression_safeguards.api.Safeguards.compute_required_stencil_for_chunked_correction]:
            - a wrapping boundary is always compatible with a valid boundary.
            - a larger stencil is always compatible with a smaller stencil.
            - a smaller stencil is sometimes compatible with a larger stencil,
              iff the smaller stencil is near the entire data boundary and
              still includes all required elements; for instance, providing the
              entire data as a single chunk with no stencil is always
              compatible with any stencil
        late_bound_chunk : Mapping[str | Parameter, Value] | Bindings
            The bindings for all late-bound parameters of the safeguards.

            The bindings must resolve all late-bound parameters and include no
            extraneous parameters.

            If a binding resolves to an array, it must be the corresponding
            chunk of the entire late-bound array.

            The safeguards automatically provide the `$x` and `$X` built-in
            constants, which must not be included.

        Returns
        -------
        chunk_ok : bool
            `True` if the check succeeded for the chunk.
        """

        assert data_chunk.dtype in _SUPPORTED_DTYPES, (
            f"can only safeguard arrays of dtype {', '.join(d.str for d in _SUPPORTED_DTYPES)}"
        )

        assert data_chunk.dtype == prediction_chunk.dtype
        assert data_chunk.shape == prediction_chunk.shape
        assert len(data_shape) == data_chunk.ndim
        assert len(chunk_offset) == data_chunk.ndim
        assert len(chunk_stencil) == data_chunk.ndim

        chunk_shape: tuple[int, ...] = tuple(
            a - s[1].before - s[1].after
            for a, s in zip(data_chunk.shape, chunk_stencil)
        )

        required_stencil = self.compute_required_stencil_for_chunked_correction(
            data_shape, smallest_chunk_shape=chunk_shape
        )

        stencil_indices: list[slice] = []
        non_stencil_indices: list[slice] = []

        # (1): check that the chunk stencil is compatible with the required stencil
        #      this is not trivial since we need to account for huge chunks where
        #       downgrading the stencil can work out
        # (2): compute indices to extract just the needed data and data+stencil
        for i, (c, r) in enumerate(zip(chunk_stencil, required_stencil)):
            # complete chunks that span the entire data along the axis are
            #  always allowed
            if (
                c[0] == BoundaryCondition.valid
                and c[1].before == 0
                and c[1].after == 0
                and chunk_shape[i] == data_shape[i]
            ):
                stencil_indices.append(slice(None))
                non_stencil_indices.append(slice(None))
                continue

            match c[0]:
                case BoundaryCondition.valid:
                    # we need to check that we requested a valid boundary,
                    #  which is only compatible with itself
                    # and that the stencil is large enough
                    assert r[0] == BoundaryCondition.valid

                    # what is the required stencil after adjusting for near-
                    #  boundary stencil truncation?
                    rs = NeighbourhoodAxis(
                        before=min(chunk_offset[i], r[1].before),
                        after=min(
                            r[1].after,
                            data_shape[i] - chunk_shape[i] - chunk_offset[i],
                        ),
                    )
                    assert c[1].before >= rs.before
                    assert c[1].after >= rs.after

                    stencil_indices.append(
                        slice(
                            c[1].before - rs.before,
                            None if c[1].after == rs.after else rs.after - c[1].after,
                        )
                    )
                    non_stencil_indices.append(
                        slice(rs.before, None if rs.after == 0 else -rs.after)
                    )
                case BoundaryCondition.wrap:
                    # a wrapping boundary is compatible with any other boundary
                    assert r[0] in (BoundaryCondition.valid, BoundaryCondition.wrap)

                    # what is the required stencil after adjusting for near-
                    #  boundary stencil truncation?
                    # (a) if the chunk is in the middle, where no boundary
                    #     condition is applied, we just keep the stencil as-is
                    # (b) if the chunk's stencil only overlaps with the boundary
                    #     on one side, we keep the stencil on that side as-is
                    # (c) otherwise, we have the full data and remove the
                    #     stencil so that the per-safeguard stencil correctly
                    #     sees how points wrap
                    rs = NeighbourhoodAxis(
                        before=(
                            r[1].before  # (a)
                            if r[1].before <= chunk_offset[i]
                            else (
                                r[1].before  # (b)
                                if r[1].after
                                <= (data_shape[i] - chunk_shape[i] - chunk_offset[i])
                                else 0  # (c)
                            )
                        ),
                        after=(
                            r[1].after  # (a)
                            if r[1].after
                            <= (data_shape[i] - chunk_shape[i] - chunk_offset[i])
                            else (
                                r[1].after  # (b)
                                if r[1].before <= chunk_offset[i]
                                else 0  # (c)
                            )
                        ),
                    )
                    assert c[1].before >= rs.before
                    assert c[1].after >= rs.after

                    stencil_indices.append(
                        slice(
                            c[1].before - rs.before,
                            None if c[1].after == rs.after else rs.after - c[1].after,
                        )
                    )
                    non_stencil_indices.append(
                        slice(rs.before, None if rs.after == 0 else -rs.after)
                    )
                case _:
                    assert_never(c[0])

        data_chunk_ = data_chunk[tuple(stencil_indices)]
        prediction_chunk_ = prediction_chunk[tuple(stencil_indices)]

        # create the late-bound bindings for the chunk and apply the stencil
        #  indices
        # TODO: check that the late-bound arrays have the same shape
        late_bound_chunk = (
            late_bound_chunk
            if isinstance(late_bound_chunk, Bindings)
            else Bindings(**late_bound_chunk)
        ).apply_slice_index(tuple(stencil_indices))

        late_bound_reqs = self.late_bound
        late_bound_builtin = {
            p: data_chunk_ for p in late_bound_reqs if p in self.builtin_late_bound
        }
        late_bound_reqs = late_bound_reqs - late_bound_builtin.keys()
        late_bound_keys = frozenset(late_bound_chunk.parameters())
        assert late_bound_reqs == late_bound_keys, (
            f"late_bound_chunk is missing bindings for {sorted(late_bound_reqs - late_bound_keys)} "
            f"/ has extraneous bindings {sorted(late_bound_keys - late_bound_reqs)}"
        )

        if len(late_bound_builtin) > 0:
            late_bound_chunk = late_bound_chunk.update(**late_bound_builtin)  # type: ignore

        # ensure we don't accidentally forget to handle new kinds of safeguards here
        assert len(self.safeguards) == len(self._pointwise_safeguards) + len(
            self._stencil_safeguards
        )

        all_ok = np.ones_like(data_chunk_[tuple(non_stencil_indices)], dtype=np.bool)

        # we need to use pointwise checks here so that we can only look at the
        #  non-stencil check results
        for safeguard in self._pointwise_safeguards + self._stencil_safeguards:
            all_ok &= safeguard.check_pointwise(
                data_chunk_, prediction_chunk_, late_bound=late_bound_chunk
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
        late_bound_chunk: Mapping[str | Parameter, Value] | Bindings = Bindings.empty(),
    ) -> np.ndarray[tuple[int, ...], np.dtype[C]]:
        """
        Compute the correction required to make the `prediction_chunk` array chunk satisfy the safeguards relative to the `data_chunk` array chunk.

        This method should only be used when working with individual chunks of
        data, otherwise please use the more efficient
        [`compute_correction`][compression_safeguards.api.Safeguards.compute_correction]
        method instead.

        Parameters
        ----------
        data_chunk : np.ndarray[S, np.dtype[T]]
            A chunk from the data array, relative to which the safeguards are
            enforced.
        prediction_chunk : np.ndarray[S, np.dtype[T]]
            The corresponding chunk from the prediction array for which the
            correction is computed.
        data_shape : tuple[int, ...]
            The shape of the entire data array, i.e. not just the chunk.
        chunk_offset : tuple[int, ...]
            The offset of the chunk inside the entire array. For arrays going
            from left to right, bottom to top, ... the offset is the index of
            the bottom left element in the entire array.
        chunk_stencil : tuple[tuple[Literal[BoundaryCondition.valid, BoundaryCondition.wrap], NeighbourhoodAxis], ...]
            The shape of the stencil neighbourhood around the chunk. This
            stencil must be compatible with the required stencil returned by
            [`compute_required_stencil_for_chunked_correction(data_shape)`][compression_safeguards.api.Safeguards.compute_required_stencil_for_chunked_correction]:
            - a wrapping boundary is always compatible with a valid boundary.
            - a larger stencil is always compatible with a smaller stencil.
            - a smaller stencil is sometimes compatible with a larger stencil,
              iff the smaller stencil is near the entire data boundary and
              still includes all required elements; for instance, providing the
              entire data as a single chunk with no stencil is always
              compatible with any stencil
        late_bound_chunk : Mapping[str | Parameter, Value] | Bindings
            The bindings for all late-bound parameters of the safeguards.

            The bindings must resolve all late-bound parameters and include no
            extraneous parameters.

            If a binding resolves to an array, it must be the corresponding
            chunk of the entire late-bound array.

            The safeguards automatically provide the `$x` and `$X` built-in
            constants, which must not be included.

        Returns
        -------
        correction_chunk : np.ndarray[tuple[int, ...], np.dtype[C]]
            The correction array chunk. The correction chunk is truncated to
            remove the stencil, i.e. it only contains the correction for the
            non-stencil-extended chunk.
        """

        assert data_chunk.dtype in _SUPPORTED_DTYPES, (
            f"can only safeguard arrays of dtype {', '.join(d.str for d in _SUPPORTED_DTYPES)}"
        )

        assert data_chunk.dtype == prediction_chunk.dtype
        assert data_chunk.shape == prediction_chunk.shape
        assert len(data_shape) == data_chunk.ndim
        assert len(chunk_offset) == data_chunk.ndim
        assert len(chunk_stencil) == data_chunk.ndim

        chunk_shape: tuple[int, ...] = tuple(
            a - s[1].before - s[1].after
            for a, s in zip(data_chunk.shape, chunk_stencil)
        )

        required_stencil = self.compute_required_stencil_for_chunked_correction(
            data_shape, smallest_chunk_shape=chunk_shape
        )

        stencil_indices: list[slice] = []
        non_stencil_indices: list[slice] = []

        # (1): check that the chunk stencil is compatible with the required stencil
        #      this is not trivial since we need to account for huge chunks where
        #       downgrading the stencil can work out
        # (2): compute indices to extract just the needed data and data+stencil
        for i, (c, r) in enumerate(zip(chunk_stencil, required_stencil)):
            # complete chunks that span the entire data along the axis are
            #  always allowed
            if (
                c[0] == BoundaryCondition.valid
                and c[1].before == 0
                and c[1].after == 0
                and chunk_shape[i] == data_shape[i]
            ):
                stencil_indices.append(slice(None))
                non_stencil_indices.append(slice(None))
                continue

            match c[0]:
                case BoundaryCondition.valid:
                    # we need to check that we requested a valid boundary,
                    #  which is only compatible with itself
                    # and that the stencil is large enough
                    assert r[0] == BoundaryCondition.valid

                    # what is the required stencil after adjusting for near-
                    #  boundary stencil truncation?
                    rs = NeighbourhoodAxis(
                        before=min(chunk_offset[i], r[1].before),
                        after=min(
                            r[1].after,
                            data_shape[i] - chunk_shape[i] - chunk_offset[i],
                        ),
                    )
                    assert c[1].before >= rs.before
                    assert c[1].after >= rs.after

                    stencil_indices.append(
                        slice(
                            c[1].before - rs.before,
                            None if c[1].after == rs.after else rs.after - c[1].after,
                        )
                    )
                    non_stencil_indices.append(
                        slice(rs.before, None if rs.after == 0 else -rs.after)
                    )
                case BoundaryCondition.wrap:
                    # a wrapping boundary is compatible with any other boundary
                    assert r[0] in (BoundaryCondition.valid, BoundaryCondition.wrap)

                    # what is the required stencil after adjusting for near-
                    #  boundary stencil truncation?
                    # (a) if the chunk is in the middle, where no boundary
                    #     condition is applied, we just keep the stencil as-is
                    # (b) if the chunk's stencil only overlaps with the boundary
                    #     on one side, we keep the stencil on that side as-is
                    # (c) otherwise, we have the full data and remove the
                    #     stencil so that the per-safeguard stencil correctly
                    #     sees how points wrap
                    rs = NeighbourhoodAxis(
                        before=(
                            r[1].before  # (a)
                            if r[1].before <= chunk_offset[i]
                            else (
                                r[1].before  # (b)
                                if r[1].after
                                <= (data_shape[i] - chunk_shape[i] - chunk_offset[i])
                                else 0  # (c)
                            )
                        ),
                        after=(
                            r[1].after  # (a)
                            if r[1].after
                            <= (data_shape[i] - chunk_shape[i] - chunk_offset[i])
                            else (
                                r[1].after  # (b)
                                if r[1].before <= chunk_offset[i]
                                else 0  # (c)
                            )
                        ),
                    )
                    assert c[1].before >= rs.before
                    assert c[1].after >= rs.after

                    stencil_indices.append(
                        slice(
                            c[1].before - rs.before,
                            None if c[1].after == rs.after else rs.after - c[1].after,
                        )
                    )
                    non_stencil_indices.append(
                        slice(rs.before, None if rs.after == 0 else -rs.after)
                    )
                case _:
                    assert_never(c[0])

        data_chunk_ = data_chunk[tuple(stencil_indices)]
        prediction_chunk_ = prediction_chunk[tuple(stencil_indices)]

        # create the late-bound bindings for the chunk and apply the stencil
        #  indices
        # TODO: check that the late-bound arrays have the same shape
        late_bound_chunk = (
            late_bound_chunk
            if isinstance(late_bound_chunk, Bindings)
            else Bindings(**late_bound_chunk)
        ).apply_slice_index(tuple(stencil_indices))

        late_bound_reqs = self.late_bound
        late_bound_builtin = {
            p: data_chunk_ for p in late_bound_reqs if p in self.builtin_late_bound
        }
        late_bound_reqs = late_bound_reqs - late_bound_builtin.keys()
        late_bound_keys = frozenset(late_bound_chunk.parameters())
        assert late_bound_reqs == late_bound_keys, (
            f"late_bound_chunk is missing bindings for {sorted(late_bound_reqs - late_bound_keys)} "
            f"/ has extraneous bindings {sorted(late_bound_keys - late_bound_reqs)}"
        )

        if len(late_bound_builtin) > 0:
            late_bound_chunk = late_bound_chunk.update(**late_bound_builtin)  # type: ignore

        # if no chunk requires a correction, this one doesn't either
        if not any_chunk_check_failed:
            return np.zeros_like(as_bits(data_chunk_)[tuple(non_stencil_indices)])

        safeguard: Safeguard

        # if only pointwise safeguards are used, check if we need to correct
        #  this chunk
        if len(self._pointwise_safeguards) == len(self.safeguards):
            all_ok = np.ones_like(
                data_chunk_[tuple(non_stencil_indices)], dtype=np.bool
            )

            # we need to use pointwise checks here so that we can only look at the
            #  non-stencil check results
            for safeguard in self._pointwise_safeguards:
                all_ok &= safeguard.check_pointwise(
                    data_chunk_, prediction_chunk_, late_bound=late_bound_chunk
                )[tuple(non_stencil_indices)]

                if not np.all(all_ok):
                    break

            if np.all(all_ok):
                return np.zeros_like(as_bits(data_chunk_)[tuple(non_stencil_indices)])

        # otherwise, correct the chunk
        # if stencil safeguards are used, then any chunk needing a correction
        #  requires all chunks to be corrected

        # ensure we don't accidentally forget to handle new kinds of safeguards here
        assert len(self.safeguards) == len(self._pointwise_safeguards) + len(
            self._stencil_safeguards
        )

        all_intervals = []
        for safeguard in self._pointwise_safeguards + self._stencil_safeguards:
            intervals = safeguard.compute_safe_intervals(
                data_chunk_, late_bound=late_bound_chunk
            )
            assert np.all(intervals.contains(data_chunk_)), (
                f"safeguard {safeguard!r}'s intervals must contain the original data"
            )
            all_intervals.append(intervals)

        combined_intervals = all_intervals[0]
        for intervals in all_intervals[1:]:
            combined_intervals = combined_intervals.intersect(intervals)
        correction_chunk = combined_intervals.pick(prediction_chunk_)

        for safeguard, intervals in zip(self.safeguards, all_intervals):
            assert np.all(intervals.contains(correction_chunk)), (
                f"safeguard {safeguard!r} interval does not contain the correction {correction_chunk!r}"
            )
            assert safeguard.check(
                data_chunk_, correction_chunk, late_bound=late_bound_chunk
            ), (
                f"safeguard {safeguard!r} check fails after correction {correction_chunk!r} on data {data_chunk_!r}"
            )

        prediction_chunk_bits = as_bits(prediction_chunk_)
        correction_chunk_bits = as_bits(correction_chunk)

        return (prediction_chunk_bits - correction_chunk_bits)[
            tuple(non_stencil_indices)
        ]

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
