"""
# Fearless (chunked) lossy compression with `xarray-safeguards`

Lossy compression can be *scary* as valuable information or features of the
data may be lost.

By using [`Safeguards`][compression_safeguards.Safeguards] to **guarantee**
your safety requirements, lossy compression can be applied safely and
*without fear*.

## Overview

This package provides functionality to use safeguards with (chunked)
[`xarray.DataArray`][xarray.DataArray]s.

In particular, `xarray-safeguards` provides the
[`produce_data_array_correction`][xarray_safeguards.produce_data_array_correction]
function to produce a correction such that certain properties of the original
data are preserved after compression, which can be stored in the same or a
different dataset (or file).

This correction can be applied to the decompressed data using the
[`apply_data_array_correction`][xarray_safeguards.apply_data_array_correction]
function or the [`.safeguarded`][xarray_safeguards.DatasetSafeguardedAccessor]
accessor on datasets.

This package also provides the
[`.safeguards`][xarray_safeguards.DataArraySafeguardsAccessor]
accessor on correction or corrected data arrays to inspect the safeguards that
were applied.

By applying the corrections produced by `produce_data_array_correction`, data
that was compressed with badly-behaving lossy compressors can be safely used,
at the cost of potentially less efficient compression, and lossy compression
can be applied *without fear*.

## Example

You can produce the corrections to uphold an absolute error bound of
$eb_{abs} = 0.1$ for an already-compressed data array as follows:

```py
import numpy as np
import xarray as xr
from xarray_safeguards import apply_data_array_correction, produce_data_array_correction

# some (chunked) n-dimensional data array
da = xr.DataArray(np.linspace(-10, 10, 21), name="da").chunk(10)
# lossy-compressed prediction for the data, here all zeros
da_prediction = xr.DataArray(np.zeros_like(da.values), name="da").chunk(10)

da_correction = produce_data_array_correction(
    data=da,
    prediction=da_prediction,
    # guarantee an absolute error bound of 0.1:
    #   |x - x'| <= 0.1
    safeguards=[dict(kind="eb", type="abs", eb=0.1)],
)

## (a) manual correction ##

da_corrected = apply_data_array_correction(da_prediction, da_correction)
np.testing.assert_allclose(da_corrected.values, da.values, rtol=0, atol=0.1)

## (b) automatic correction with xarray accessors ##

# combine the lossy prediction and the correction into one dataset
#  e.g. by loading them from different files using `xarray.open_mfdataset`
ds = xr.Dataset({
    da_prediction.name: da_prediction,
    da_correction.name: da_correction,
})

# access the safeguarded dataset that applies all corrections
ds_safeguarded: xr.Dataset = ds.safeguarded
np.testing.assert_allclose(ds_safeguarded["da"].values, da.values, rtol=0, atol=0.1)
```

Please refer to the
[`compression_safeguards.SafeguardKind`][compression_safeguards.SafeguardKind]
for an enumeration of all supported safeguards.
"""

__all__ = [
    "produce_data_array_correction",
    "apply_data_array_correction",
    "DataValue",
    "DatasetSafeguardedAccessor",
    "DataArraySafeguardsAccessor",
]

import json
import warnings
from collections.abc import Collection, Mapping
from types import MappingProxyType
from typing import Literal, TypeAlias

import dask
import dask.array
import numpy as np
import xarray as xr
from compression_safeguards.api import Safeguards
from compression_safeguards.safeguards.abc import Safeguard
from compression_safeguards.safeguards.stencil import (
    BoundaryCondition,
    NeighbourhoodAxis,
)
from compression_safeguards.utils.bindings import Parameter, Value
from typing_extensions import assert_never  # MSPV 3.11

DataValue: TypeAlias = int | float | np.number | xr.DataArray
"""
Parameter value type that includes scalar numbers and arrays thereof.
"""


def produce_data_array_correction(
    data: xr.DataArray,
    prediction: xr.DataArray,
    safeguards: Collection[dict | Safeguard],
    late_bound: Mapping[str, DataValue] = MappingProxyType(dict()),
    *,
    allow_unsafe_safeguards_override: bool = False,
) -> xr.DataArray:
    """
    Produce the correction required to make the `prediction` data array satisfy the `safeguards` relative to the `data` array.

    The `data` array may be chunked[^1] and the `prediction` array must use the
    same chunking. Importantly, the `data` array must contain the complete data,
    i.e. not just a sub-chunk of the data, so that non-pointwise safeguards are
    correctly applied.

    If the the `data` array is chunked, the correction is produced lazily,
    otherwise it is computed eagerly.

    [^1]: At the moment, only chunking with `dask` is supported.

    Parameters
    ----------
    data : xr.DataArray
        The data array, relative to which the safeguards are enforced.
    prediction : xr.DataArray
        The prediction array for which the correction is produced.
    safeguards : Collection[dict | Safeguard]
        The safeguards that will be applied relative to the `data` array.

        They can either be
        passed as a safeguard configuration [`dict`][dict] or an already
        initialized
        [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard].

        Please refer to the
        [`SafeguardKind`][compression_safeguards.safeguards.SafeguardKind]
        for an enumeration of all supported safeguards.
    late_bound : Mapping[str, DataValue]
        The bindings for all late-bound parameters of the `safeguards`.

        The bindings must resolve all late-bound parameters and include no
        extraneous parameters.

        If a binding resolves to a [`xr.DataArray`][xarray.DataArray], its
        dimensions must be a subset of `data.dims` and the shape must be
        broadcastable to the data shape, i.e. for each axis must either have
        size 1 or the same size as the matching dimension in the `data`.

        This method automatically provides the following built-in constants
        to the safeguards, which must not be included:

        - `$x` and `$X`: the original `data` as a constant
        - `$x_min` and `$x_max`: the global minimum/maximum of the data
        - `$d_DIM` for each dimension `DIM` of the `data` array
    allow_unsafe_safeguards_override : bool
        **WARNING:** This option is *unsafe* and must only be passed if
        instructed. Do *not* pass this option otherwise.

    Returns
    -------
    correction : xr.DataArray
        The correction array
    """

    # small safeguard against the printer problem
    assert "safeguards" not in data.attrs, (
        "computing the safeguards correction relative to a `data` array that "
        "has *already* been safeguards-corrected before is unsafe as "
        "compression errors can accumulate when the original uncompressed data "
        "is not known; this is also known as the printer problem; please pass "
        "the original uncompressed and uncorrected `data` to ensure that the "
        "safeguards can be applied correctly"
    )

    if "safeguards" in prediction.attrs:
        if allow_unsafe_safeguards_override:
            warnings.warn("`allow_unsafe_safeguards_override`=True")
        else:
            assert "safeguards" not in prediction.attrs, (
                "computing the safeguards correction for a `prediction` array "
                "that has *already* been safeguards-corrected before is unsafe "
                "as the safety guarantees provided by the previously applied "
                "safeguards may not be provided by the newly applied "
                "safeguards, thus violating the earlier guarantees; this is "
                "also known as the printer problem; please manually inspect "
                "the `.safeguards` property of the `prediction` array and "
                "ensure that they are included in the new `safeguards` and "
                "then pass `allow_unsafe_safeguards_override=True` to this "
                "function"
            )
    else:
        assert allow_unsafe_safeguards_override is False

    assert data.dims == prediction.dims, "data.dims != prediction.dims"
    assert data.shape == prediction.shape, "data.shape != prediction.shape"
    assert data.dtype == prediction.dtype, "data.dtype != prediction.dtype"
    assert data.chunks == prediction.chunks, "data.chunks != prediction.chunks"

    safeguards_: Safeguards = Safeguards(safeguards=safeguards)

    builtin_late_bound: frozenset[Parameter] = frozenset(
        safeguards_.builtin_late_bound
    ) | frozenset(
        [Parameter("$x_min"), Parameter("$x_max")]
        + [Parameter(f"$d_{d}") for d in data.dims]
    )

    safeguards_late_bound_reqs = frozenset(safeguards_.late_bound)
    late_bound_reqs = frozenset(safeguards_late_bound_reqs - builtin_late_bound)
    late_bound_keys = frozenset(Parameter(k) for k in late_bound.keys())

    assert late_bound_reqs == late_bound_keys, (
        f"late_bound is missing bindings for {sorted(late_bound_reqs - late_bound_keys)} "
        f"/ has extraneous bindings {sorted(late_bound_keys - late_bound_reqs)}"
    )

    # create the global built-in late-bound bindings with $x_min and $x_max
    #  and split-out the late-bound data array bindings that require chunking
    late_bound_global: dict[str, int | float | np.number] = dict()
    late_bound_data_arrays: dict[str, xr.DataArray] = dict()
    if "$x_min" in safeguards_late_bound_reqs:
        da_min = (
            np.nanmin(data)
            if data.size > 0 and not np.all(np.isnan(data))
            else np.array(0, dtype=data.dtype)
        )
        late_bound_global["$x_min"] = da_min  # type: ignore
    if "$x_max" in safeguards_late_bound_reqs:
        da_max = (
            np.nanmax(data)
            if data.size > 0 and not np.all(np.isnan(data))
            else np.array(0, dtype=data.dtype)
        )
        late_bound_global["$x_max"] = da_max  # type: ignore
    for k, v in late_bound.items():
        if isinstance(v, int | float | np.number):
            late_bound_global[k] = v
        else:
            assert isinstance(v, xr.DataArray), (
                f"late-bound param {k} must be a scalar or `xarray.DataArray`"
            )
            assert frozenset(v.dims).issubset(data.dims), (
                f"late-bound param {k} dims are not a subset of data dims"
            )
            for d, ds in v.sizes.items():
                assert (ds == 1) or (ds == data.sizes[d]), (
                    "late-bound param {k} dim {d} is not broadcastable to the data size"
                )
            late_bound_data_arrays[k] = v

    correction_name = "sg" if data.name is None else f"{data.name}_sg"
    correction_attrs = dict(
        safeguarded=data.name, safeguards=json.dumps(safeguards_.get_config())
    )

    da_correction: xr.DataArray

    # special case for no chunking: just compute eagerly
    if data.chunks is None:
        # provide built-in late-bound bindings $d for the data dimensions
        late_bound_chunk: dict[str, Value] = dict(**late_bound_global)
        for i, d in enumerate(data.dims):
            if f"$d_{d}" in safeguards_late_bound_reqs:
                shape = [1 for _ in range(len(data.dims))]
                shape[i] = data.shape[i]
                late_bound_chunk[f"$d_{d}"] = data[d].values.reshape(shape)
        for k, dv in late_bound_data_arrays.items():
            axes = sorted(
                [i for i in range(len(dv.dims))],
                key=lambda i: data.dims.index(dv.dims[i]),
            )
            shape = [1 for _ in range(len(data.dims))]
            for d in dv.dims:
                i = data.dims.index(d)
                shape[i] = data.shape[i]
            late_bound_chunk[k] = dv.values.transpose(axes).reshape(shape)

        da_correction = (
            data.copy(
                data=safeguards_.compute_correction(
                    data.values, prediction.values, late_bound=late_bound_chunk
                )
            )
            .rename(correction_name)
            .assign_attrs(**correction_attrs)
        )

        # drop the previous encoding from the variable but not the coords
        return da_correction._replace(variable=da_correction.variable.drop_encoding())

    # provide built-in late-bound bindings $d for the data dimensions
    chunked_late_bound: dict[str, dask.array.Array] = dict()
    for i, d in enumerate(data.dims):
        if f"$d_{d}" in safeguards_late_bound_reqs:
            shape = [1 for _ in range(len(data.dims))]
            shape[i] = data.shape[i]
            chunked_late_bound[f"$d_{d}"] = dask.array.broadcast_to(
                data[d].data,
                shape=shape,
                meta=np.array((), dtype=data[d].dtype),
            ).rechunk(data.chunks)
    for k, v in late_bound_data_arrays.items():
        dims = sorted(v.dims, key=lambda i: data.dims.index(i))
        shape = [1 for _ in range(len(data.dims))]
        for d in v.dims:
            i = data.dims.index(d)
            shape[i] = data.shape[i]
        chunked_late_bound[k] = dask.array.broadcast_to(
            v.transpose(*dims, transpose_coords=False).data,
            shape=shape,
            meta=np.array((), dtype=v.dtype),
        ).rechunk(data.chunks)

    required_stencil = safeguards_.compute_required_stencil_for_chunked_correction(
        data.shape
    )
    correction_dtype: np.dtype = safeguards_.correction_dtype_for_data(data.dtype)

    # special case for no stencil: just apply independently to each chunk
    if all(s.before == 0 and s.after == 0 for b, s in required_stencil):

        def _compute_independent_chunk_correction(
            data_chunk: np.ndarray,
            prediction_chunk: np.ndarray,
            *late_bound_chunks: np.ndarray,
            late_bound_names: tuple[str],
            late_bound_global: dict[str, int | float | np.number],
            safeguards: Safeguards,
        ) -> np.ndarray:
            assert len(late_bound_chunks) == len(late_bound_names), (
                "late-bound chunks and names mismatch"
            )

            late_bound_chunk: dict[str, Value] = dict(
                **late_bound_global,
                **{p: v for p, v in zip(late_bound_names, late_bound_chunks)},
            )

            return safeguards.compute_correction(
                data_chunk, prediction_chunk, late_bound=late_bound_chunk
            )

        da_correction = (
            data.copy(
                data=data.data.map_blocks(
                    _compute_independent_chunk_correction,
                    prediction.data,
                    *chunked_late_bound.values(),
                    dtype=correction_dtype,
                    chunks=None,
                    enforce_ndim=True,
                    meta=np.array((), dtype=correction_dtype),
                    late_bound_names=tuple(chunked_late_bound.keys()),
                    late_bound_global=late_bound_global,
                    safeguards=safeguards_,
                )
            )
            .rename(correction_name)
            .assign_attrs(**correction_attrs)
        )

        # drop the previous encoding from the variable but not the coords
        return da_correction._replace(variable=da_correction.variable.drop_encoding())

    boundary: Literal["none", "periodic"] = "none"
    depth_: list[tuple[int, int]] = []
    for a, (b, s) in zip(data.shape, required_stencil):
        # dask doesn't support depths larger than the axes,
        # so clip that axis and prefer no boundary condition
        #  as it will anyways be rechunked to just a single chunk
        if s.before >= a or s.after >= a:
            depth_.append((a, a))
        else:
            depth_.append((s.before, s.after))
            match b:
                case BoundaryCondition.valid:
                    pass
                case BoundaryCondition.wrap:
                    boundary = "periodic"
                case _:
                    assert_never(b)

    match boundary:
        case "none":
            depth: tuple[int | tuple[int, int], ...] = tuple(
                b if b == a else (a, b) for a, b in depth_
            )
        case "periodic":
            # dask only supports asymmetric depths for the none boundary
            depth = tuple(max(a, b) for a, b in depth_)
        case _:
            assert_never(boundary)

    def _check_overlapping_stencil_chunk(
        data_chunk: np.ndarray,
        prediction_chunk: np.ndarray,
        data_indices_chunk: np.ndarray,
        *late_bound_chunks: np.ndarray,
        late_bound_names: tuple[str],
        late_bound_global: dict[str, int | float | np.number],
        safeguards: Safeguards,
        data_shape: tuple[int, ...],
        depth_: tuple[int | tuple[int, int], ...],
        boundary_: Literal["none", "periodic"],
        block_info=None,
    ) -> np.ndarray:
        assert block_info is not None, "missing block info"
        assert len(late_bound_chunks) == len(late_bound_names), (
            "late-bound chunks and names mismatch"
        )
        assert len(depth_) == data_chunk.ndim, "overlap depth length mismatch"

        late_bound_chunk: dict[str, Value] = dict(
            **late_bound_global,
            **{p: v for p, v in zip(late_bound_names, late_bound_chunks)},
        )

        depth: tuple[tuple[int, int], ...] = tuple(
            (d, d) if isinstance(d, int) else d for d in depth_
        )

        match boundary_:
            case "none":
                # map_overlap passes the wrong block_info,
                #  see https://github.com/dask/dask/issues/11349
                # so we only trust it just enough to compute the effective
                #  stencil size
                extended_data_shape: tuple[int, ...] = tuple(block_info[None]["shape"])
                extended_chunk_location: tuple[tuple[int, int], ...] = tuple(
                    block_info[None]["array-location"]
                )

                # the none boundary does not extend beyond the data boundary,
                #  so we need to check by how much the stencil was cut off
                chunk_stencil: tuple[
                    tuple[
                        Literal[BoundaryCondition.valid, BoundaryCondition.wrap],
                        NeighbourhoodAxis,
                    ],
                    ...,
                ] = tuple(
                    (
                        BoundaryCondition.valid,
                        NeighbourhoodAxis(
                            before=s - max(0, s - b), after=min(e + a, d) - e
                        ),
                    )
                    for (b, a), (s, e), d in zip(
                        depth, extended_chunk_location, extended_data_shape
                    )
                )
            case "periodic":
                # the periodic boundary guarantees that we always have the
                #  stencil we asked for
                chunk_stencil = tuple(
                    (BoundaryCondition.wrap, NeighbourhoodAxis(before=b, after=a))
                    for b, a in depth
                )
            case _:
                assert_never(boundary_)

        # extract the indices of the non-stencil-extended data indices chunk
        data_indices_chunk = data_indices_chunk[
            tuple(
                slice(a.before, None if a.after == 0 else -a.after)
                for b, a in chunk_stencil
            )
        ]
        chunk_shape = data_indices_chunk.shape

        # extract the offset of the chunk from the data indices chunk
        if data_indices_chunk.size > 0:
            chunk_offset_index = int(data_indices_chunk.flatten()[0])
            chunk_offset_ = []
            for s in data_shape[::-1]:
                chunk_offset_.append(chunk_offset_index % s)
                chunk_offset_index //= s
            chunk_offset: tuple[int, ...] = tuple(chunk_offset_[::-1])
        else:
            chunk_offset = tuple(0 for _ in data_shape)

        # this is safe because
        # - map_overlap ensures we get chunks including their required stencil
        chunk_is_ok = safeguards.check_chunk(
            data_chunk,
            prediction_chunk,
            data_shape=data_shape,
            chunk_offset=chunk_offset,
            chunk_stencil=chunk_stencil,
            late_bound_chunk=late_bound_chunk,
        )

        # broadcast the boolean check scalar to the output chunk shape
        return np.broadcast_to(chunk_is_ok, chunk_shape)

    data_indices = dask.array.arange(data.size).reshape(data.shape).rechunk(data.chunks)

    # first check each chunk to find out if any failed the check
    # for stencil safeguards, one chunk failing requires all to be corrected
    #  (since incremental corrections could spread)
    # if no chunk check failed, the chunked correction can take a fast path
    any_chunk_check_failed = (
        not dask.array.map_overlap(
            _check_overlapping_stencil_chunk,
            data.data,
            prediction.data,
            data_indices,
            *chunked_late_bound.values(),
            dtype=np.bool,
            # we cannot output 1x...x1 chunks here since we later use the block
            #  info to compute the data size and chunk location
            chunks=None,
            enforce_ndim=True,
            meta=np.array((), dtype=np.bool),
            depth=depth,
            boundary=boundary,
            trim=False,
            align_arrays=False,
            # if the stencil is larger than the smallest chunk, temporary rechunking may be necessary
            allow_rechunk=True,
            late_bound_names=tuple(chunked_late_bound.keys()),
            late_bound_global=late_bound_global,
            safeguards=safeguards_,
            data_shape=data.shape,
            depth_=depth,
            boundary_=boundary,
        )
        .all()
        .compute()
    )

    def _compute_overlapping_stencil_chunk_correction(
        data_chunk: np.ndarray,
        prediction_chunk: np.ndarray,
        data_indices_chunk: np.ndarray,
        *late_bound_chunks: np.ndarray,
        late_bound_names: tuple[str],
        late_bound_global: dict[str, int | float | np.number],
        safeguards: Safeguards,
        data_shape: tuple[int, ...],
        depth_: tuple[int | tuple[int, int], ...],
        boundary_: Literal["none", "periodic"],
        any_chunk_check_failed: bool,
        block_info=None,
    ) -> np.ndarray:
        assert block_info is not None, "missing block info"
        assert len(late_bound_chunks) == len(late_bound_names), (
            "late-bound chunks and names mismatch"
        )
        assert len(depth_) == data_chunk.ndim, "overlap depth length mismatch"

        late_bound_chunk: dict[str, Value] = dict(
            **late_bound_global,
            **{p: v for p, v in zip(late_bound_names, late_bound_chunks)},
        )

        depth: tuple[tuple[int, int], ...] = tuple(
            (d, d) if isinstance(d, int) else d for d in depth_
        )

        match boundary_:
            case "none":
                # map_overlap passes the wrong block_info,
                #  see https://github.com/dask/dask/issues/11349
                # so we only trust it just enough to compute the effective
                #  stencil size
                extended_data_shape: tuple[int, ...] = tuple(block_info[None]["shape"])
                extended_chunk_location: tuple[tuple[int, int], ...] = tuple(
                    block_info[None]["array-location"]
                )

                # the none boundary does not extend beyond the data boundary,
                #  so we need to check by how much the stencil was cut off
                chunk_stencil: tuple[
                    tuple[
                        Literal[BoundaryCondition.valid, BoundaryCondition.wrap],
                        NeighbourhoodAxis,
                    ],
                    ...,
                ] = tuple(
                    (
                        BoundaryCondition.valid,
                        NeighbourhoodAxis(
                            before=s - max(0, s - b), after=min(e + a, d) - e
                        ),
                    )
                    for (b, a), (s, e), d in zip(
                        depth, extended_chunk_location, extended_data_shape
                    )
                )
            case "periodic":
                # the periodic boundary guarantees that we always have the
                #  stencil we asked for
                chunk_stencil = tuple(
                    (BoundaryCondition.wrap, NeighbourhoodAxis(before=b, after=a))
                    for b, a in depth
                )
            case _:
                assert_never(boundary_)

        # extract the indices of the non-stencil-extended data indices chunk
        data_indices_chunk = data_indices_chunk[
            tuple(
                slice(a.before, None if a.after == 0 else -a.after)
                for b, a in chunk_stencil
            )
        ]
        chunk_shape = data_indices_chunk.shape

        # extract the offset of the chunk from the data indices chunk
        if data_indices_chunk.size > 0:
            chunk_offset_index = int(data_indices_chunk.flatten()[0])
            chunk_offset_ = []
            for s in data_shape[::-1]:
                chunk_offset_.append(chunk_offset_index % s)
                chunk_offset_index //= s
            chunk_offset: tuple[int, ...] = tuple(chunk_offset_[::-1])
        else:
            chunk_offset = tuple(0 for _ in data_shape)

        # this is safe because
        # - map_overlap ensures we get chunks including their required stencil
        # - compute_chunked_correction only returns the correction for the non-
        #   overlapping non-stencil parts of the chunk
        correction: np.ndarray = safeguards.compute_chunked_correction(
            data_chunk,
            prediction_chunk,
            data_shape=data_shape,
            chunk_offset=chunk_offset,
            chunk_stencil=chunk_stencil,
            any_chunk_check_failed=any_chunk_check_failed,
            late_bound_chunk=late_bound_chunk,
        )
        assert correction.shape == chunk_shape, "invalid correction chunk shape"

        return correction

    da_correction = (
        data.copy(
            data=dask.array.map_overlap(
                _compute_overlapping_stencil_chunk_correction,
                data.data,
                prediction.data,
                data_indices,
                *chunked_late_bound.values(),
                dtype=correction_dtype,
                chunks=None,
                enforce_ndim=True,
                meta=np.array((), dtype=correction_dtype),
                depth=depth,
                boundary=boundary,
                trim=False,
                align_arrays=False,
                # if the stencil is larger than the smallest chunk, temporary rechunking may be necessary
                allow_rechunk=True,
                late_bound_names=tuple(chunked_late_bound.keys()),
                late_bound_global=late_bound_global,
                safeguards=safeguards_,
                data_shape=data.shape,
                depth_=depth,
                boundary_=boundary,
                any_chunk_check_failed=any_chunk_check_failed,
            )
            .compute_chunk_sizes()  # shape and chunk size may be wrong, recompute
            .rechunk(data.chunks)  # undo temporary rechunking
        )
        .rename(correction_name)
        .assign_attrs(**correction_attrs)
    )

    # drop the previous encoding from the variable but not the coords
    return da_correction._replace(variable=da_correction.variable.drop_encoding())


def apply_data_array_correction(
    prediction: xr.DataArray,
    correction: xr.DataArray,
) -> xr.DataArray:
    """
    Apply the `correction` to the `prediction` array to satisfy the safeguards for which the `correction` was produced.

    The `prediction` and `correction` arrays must use the same chunking, though
    this chunking may differ from the one that was used to produce the
    `correction`.

    If the the `prediction` array is chunked, the correction is applied lazily,
    otherwise it its application is computed eagerly.

    Parameters
    ----------
    prediction : xr.DataArray
        The prediction array for which the correction has been produced.
    correction : xr.DataArray
        The correction array.

    Returns
    -------
    corrected : xr.DataArray
        The corrected array, which satisfies the safeguards.
    """

    assert correction.dims == prediction.dims, "correction.dims != prediction.dims"
    assert correction.shape == prediction.shape, "correction.shape != prediction.shape"
    assert correction.chunks == prediction.chunks, (
        "correction.chunks != prediction.chunks"
    )

    assert "safeguards" in correction.attrs, (
        "correction does not contain metadata about the safeguards it was produced with"
    )
    safeguards = Safeguards.from_config(json.loads(correction.attrs["safeguards"]))

    assert correction.dtype == safeguards.correction_dtype_for_data(prediction.dtype), (
        "correction dtype mismatch"
    )

    def _apply_independent_chunk_correction(
        prediction_chunk: xr.DataArray,
        correction_chunk: xr.DataArray,
        safeguards: Safeguards,
    ) -> xr.DataArray:
        return prediction_chunk.copy(
            data=safeguards.apply_correction(
                prediction_chunk.values, correction_chunk.values
            )
        )

    return xr.map_blocks(
        _apply_independent_chunk_correction,
        prediction,
        args=(correction,),
        kwargs=dict(safeguards=safeguards),
        template=prediction,
    ).assign_attrs(safeguards=correction.attrs["safeguards"])


@xr.register_dataset_accessor("safeguarded")
class DatasetSafeguardedAccessor:
    """
    An extension for an [`xarray.Dataset`][xarray.Dataset] that provides the
    `.safeguarded` property that applies safeguards corrections in the dataset
    to their respective variables.

    For instance, for a dataset `ds` that contains both a variable `ds.foo` and
    its correction, you can access the corrected variable using
    `ds.safeguarded.foo`.
    """

    __slots__ = ()

    def __new__(cls, ds: xr.Dataset) -> xr.Dataset:  # type: ignore
        corrected: dict[str, xr.DataArray] = dict()

        for v in ds.data_vars.values():
            if "safeguarded" in v.attrs:
                kp = v.attrs["safeguarded"]
                assert kp is not None, "safeguarded attr must not be None"
                assert kp in ds.data_vars, (
                    "safeguarded attr must refer to another variable"
                )
                assert "safeguards" in v.attrs, "safeguards attr must exist"

                corrected[kp] = apply_data_array_correction(ds.data_vars[kp], v)

        if len(corrected) == 0:
            raise AttributeError(
                "not a dataset with not-yet-applied safeguards corrections"
            )

        return xr.Dataset(corrected, attrs=ds.attrs)


@xr.register_dataarray_accessor("safeguards")
class DataArraySafeguardsAccessor:
    """
    An extension for an [`xarray.DataArray`][xarray.DataArray] that provides
    the `.safeguards` property that exposes the collection of
    [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard]s for
    a safeguards correction or corrected array.
    """

    __slots__ = ()

    def __new__(cls, da: xr.DataArray) -> Collection[Safeguard]:  # type: ignore
        if "safeguards" not in da.attrs:
            raise AttributeError("not a data array with safeguards")
        return Safeguards.from_config(json.loads(da.attrs["safeguards"])).safeguards
